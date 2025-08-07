"""
BART 기반 대화 요약 모델 학습 스크립트
baseline.ipynb와 config.yaml을 기반으로 구현
+ sweep.yaml 하이퍼파라미터 최적화 지원
"""

import os
import sys
import yaml
import torch
import argparse
from rouge import Rouge
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    BartForConditionalGeneration, 
    BartConfig,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
import random, numpy as np
from transformers import set_seed as hf_set_seed

# 현재 파일 위치 기준으로 data 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "data"))

# 이제 src 경로가 sys.path 에 등록되었으므로 utils 를 안전하게 import
from utils.postprocess import postprocess
from utils.metrics import calculate_rouge_scores


def set_all_seeds(seed: int):
    """random / numpy / torch / HF transformers 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def load_config_with_overrides(config_path=None, overrides=None):
    """설정 파일 로드 및 sweep 파라미터 오버라이드"""
    # Config 파일 경로 설정
    if config_path is None:
        config_path = os.path.join(project_root, "src", "config", "config.yaml")
    
    print(f"📁 설정 파일 로드: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # WandB sweep에서 전달된 파라미터들로 config 오버라이드
    if overrides:
        print(f"🔄 Sweep 파라미터 오버라이드: {overrides}")
        
        # training 섹션 파라미터들
        if 'learning_rate' in overrides:
            config['training']['learning_rate'] = overrides['learning_rate']
        if 'num_train_epochs' in overrides:
            config['training']['num_train_epochs'] = overrides['num_train_epochs']
        if 'weight_decay' in overrides:
            config['training']['weight_decay'] = overrides['weight_decay']
        if 'per_device_train_batch_size' in overrides:
            config['training']['per_device_train_batch_size'] = overrides['per_device_train_batch_size']
        if 'gradient_accumulation_steps' in overrides:
            config['training']['gradient_accumulation_steps'] = overrides['gradient_accumulation_steps']
            
        # inference 섹션 파라미터들
        if 'num_beams' in overrides:
            config['inference']['num_beams'] = overrides['num_beams']
            
        # 모델 dropout (모델 config에 적용)
        if 'dropout' in overrides:
            config['model_overrides'] = config.get('model_overrides', {})
            config['model_overrides']['dropout'] = overrides['dropout']
            
        # 추가 training 파라미터들
        if 'warmup_ratio' in overrides:
            config['training']['warmup_ratio'] = overrides['warmup_ratio']
        if 'label_smoothing' in overrides:
            config['training']['label_smoothing_factor'] = overrides['label_smoothing']
        if 'length_penalty' in overrides:
            config['inference']['length_penalty'] = overrides['length_penalty']
        if 'repetition_penalty' in overrides:
            config['inference']['repetition_penalty'] = overrides['repetition_penalty']
        if 'generation_max_length' in overrides:
            config['training']['generation_max_length'] = overrides['generation_max_length']
    
    return config


def compute_metrics(config, tokenizer, pred):
    """평가 지표 계산 함수"""
    predictions, labels = pred.predictions, pred.label_ids

    # 디코딩
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 후처리
    remove_tokens = config['inference']['remove_tokens']
    postprocessed_preds = [postprocess(pred, remove_tokens) for pred in decoded_preds]
    postprocessed_labels = [[label] for label in decoded_labels]

    # 새로운 metric 함수를 사용하여 점수 계산
    result = calculate_rouge_scores(postprocessed_preds, postprocessed_labels)

    # 로그 출력
    print("\n" + "="*10, "ROUGE Scores", "="*10)
    print(result)
    print("="*35 + "\n")

    return result


def prepare_train_dataset(config, data_path, tokenizer):
    """학습 데이터셋 준비"""
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    # CSV 파일을 pandas DataFrame으로 로드
    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)

    # pandas DataFrame을 Hugging Face Dataset 객체로 변환
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print('-' * 150)
    print(f'train_data:\n {train_dataset[0]["dialogue"]}')
    print(f'train_label:\n {train_dataset[0]["summary"]}')
    print('-' * 150)
    print(f'val_data:\n {val_dataset[0]["dialogue"]}')
    print(f'val_label:\n {val_dataset[0]["summary"]}')

    def tokenize_function(examples):
        # 모델 입력(인코더) 토크나이징
        model_inputs = tokenizer(
            examples['dialogue'],
            max_length=config['tokenizer']['encoder_max_len'],
            truncation=True,
            padding='max_length'
        )
        
        # 레이블(디코더 출력) 토크나이징
        labels = tokenizer(
            text_target=examples['summary'],
            max_length=config['tokenizer']['decoder_max_len'],
            truncation=True,
            padding='max_length'
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # map 함수를 사용하여 데이터셋 전체에 토크나이징 적용
    # batched=True로 설정하여 여러 샘플을 한 번에 처리해 속도 향상
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # 불필요한 컬럼 제거
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(train_dataset.column_names)
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(val_dataset.column_names)

    print('-' * 10, 'Make dataset complete', '-' * 10)
    return tokenized_train_dataset, tokenized_val_dataset


def load_tokenizer_and_model_for_train(config, device):
    """토크나이저와 모델 로드"""
    print('-' * 10, 'Load tokenizer & model', '-' * 10)
    print('-' * 10, f'Model Name : {config["general"]["model_name"]}', '-' * 10)
    
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    
    # sweep에서 지정된 dropout 값이 있으면 적용
    if 'model_overrides' in config and 'dropout' in config['model_overrides']:
        bart_config.dropout = config['model_overrides']['dropout']
        bart_config.attention_dropout = config['model_overrides']['dropout']
        print(f"🎯 Dropout 설정: {config['model_overrides']['dropout']}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(
        config['general']['model_name'], 
        config=bart_config
    )

    # 특수 토큰 추가
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # 모델 크기 조정 및 디바이스 이동
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    
    print(f"Model config: {generate_model.config}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)
    
    return generate_model, tokenizer


def setup_training_arguments(config):
    """학습 인수 설정"""
    print('-' * 10, 'Make training arguments', '-' * 10)
    
    # 출력 디렉토리 생성
    output_dir = config['general']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['tokenizer']['decoder_max_len'],  # decoder_max_len과 통일
        generation_num_beams=config['inference']['num_beams'],  # sweep에서 조정 가능
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to'],
        metric_for_best_model='final_score',
        greater_is_better=True,  # final_score가 높을수록 좋은 모델
        save_safetensors=True,  # safetensors 형태로 저장
        label_smoothing_factor=config['training'].get('label_smoothing_factor', 0.0),
    )
    
    print('-' * 10, 'Make training arguments complete', '-' * 10)
    return training_args


def setup_wandb(config, is_sweep=False):
    """WandB 초기화 (비활성화됨)"""
    pass


def create_trainer(config, model, tokenizer, train_dataset, val_dataset, training_args):
    """Trainer 생성"""
    print('-' * 10, 'Make trainer', '-' * 10)

    # EarlyStopping 콜백
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    # Trainer 클래스 정의
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping_callback]
    )
    
    print('-' * 10, 'Make trainer complete', '-' * 10)
    return trainer


def save_final_model(config, trainer, tokenizer):
    """최종 모델과 토크나이저 저장"""
    print("🚀 최종 모델 저장 시작...")
    
    # 타임스탬프를 포함한 모델 경로 설정
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_model_dir = f"model_{timestamp}"
    final_model_path = os.path.join(config['general']['output_dir'], timestamped_model_dir)
    final_model_abs_path = os.path.abspath(final_model_path)
    os.makedirs(final_model_abs_path, exist_ok=True)
    
    # 'latest' 심볼릭 링크 생성 (가장 최근 모델)
    latest_link_path = os.path.join(config['general']['output_dir'], "latest")
    latest_link_abs_path = os.path.abspath(latest_link_path)
    
    # 기존 latest 링크가 있다면 제거
    if os.path.exists(latest_link_abs_path) or os.path.islink(latest_link_abs_path):
        os.remove(latest_link_abs_path)
    
    # 새로운 latest 링크 생성
    os.symlink(timestamped_model_dir, latest_link_abs_path)
    
    # 모델과 토크나이저 저장
    trainer.save_model(final_model_abs_path)
    tokenizer.save_pretrained(final_model_abs_path)
    
    # 설정 파일도 함께 저장
    config_save_path = os.path.join(final_model_abs_path, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 모델 저장: {timestamped_model_dir}")
    print(f"🔗 latest → {timestamped_model_dir}")
    
    return final_model_abs_path, timestamped_model_dir


def update_best_model(config, current_model_path, model_dir_name, trainer):
    """현재 모델의 성능을 평가하여 최고 성능 모델 업데이트"""
    try:
        # 현재 모델 평가
        current_metrics = trainer.evaluate()
        current_score = current_metrics.get('eval_rouge_1', 0.0)
        
        print(f"[INFO] 현재 모델 점수: {current_score:.4f}, 최고 점수: 업데이트 예정")
        
        # 성능 기록 파일 경로
        models_dir = config['general']['output_dir']
        best_score_file = os.path.join(models_dir, "best_score.txt")
        best_model_info_file = os.path.join(models_dir, "best_model_info.txt")
        best_link_path = os.path.join(models_dir, "best")
        best_link_abs_path = os.path.abspath(best_link_path)
        
        # 이전 최고 점수 읽기
        best_score = 0.0
        if os.path.exists(best_score_file):
            try:
                with open(best_score_file, 'r') as f:
                    best_score = float(f.read().strip())
            except:
                best_score = 0.0
        
        # 현재 모델이 더 좋은 성능을 보이면 업데이트
        if current_score > best_score:
            print(f"🏆 NEW BEST: {current_score:.4f} (이전: {best_score:.4f})")
            
            # 최고 점수 저장
            with open(best_score_file, 'w') as f:
                f.write(str(current_score))
            
            # 최고 모델 정보 저장
            with open(best_model_info_file, 'w') as f:
                f.write(f"model_dir: {model_dir_name}\n")
                f.write(f"rouge_1: {current_score:.4f}\n")
                f.write(f"timestamp: {model_dir_name.replace('model_', '')}\n")
            
            # 기존 best 링크 제거
            if os.path.exists(best_link_abs_path) or os.path.islink(best_link_abs_path):
                os.remove(best_link_abs_path)
            
            # 새로운 best 링크 생성
            os.symlink(model_dir_name, best_link_abs_path)
            print(f"🔗 best → {model_dir_name}")
            
        else:
            if os.path.exists(best_link_abs_path):
                current_best = os.readlink(best_link_abs_path)
                print(f"[INFO] 현재 모델 점수: {current_score:.4f}, 최고 점수: {best_score:.4f}")
                print("최고스코어의 업데이트는 없었습니다.")
            else:
                # best 링크가 없으면 현재 모델을 best로 설정
                print(f"🏆 FIRST MODEL: {current_score:.4f}")
                with open(best_score_file, 'w') as f:
                    f.write(str(current_score))
                with open(best_model_info_file, 'w') as f:
                    f.write(f"model_dir: {model_dir_name}\n")
                    f.write(f"rouge_1: {current_score:.4f}\n")
                    f.write(f"timestamp: {model_dir_name.replace('model_', '')}\n")
                os.symlink(model_dir_name, best_link_abs_path)
                print(f"🔗 best → {model_dir_name}")
        
    except Exception as e:
        print(f"❌ 성능 평가 실패: {e}")


def main(config_path=None, sweep_config=None):
    """메인 학습 함수"""
    print("🚀 [1/8] 설정 파일 로드 시작...")
    
    # sweep 파라미터 확인
    is_sweep = sweep_config is not None
    
    # 설정 로드
    config = load_config_with_overrides(config_path, sweep_config)
    print("✅ [1/8] 설정 파일 로드 완료")

    # 전역 시드 고정
    seed_val = config.get('general', {}).get('seed', config['training'].get('seed', 42))
    set_all_seeds(seed_val)
    print(f"🔒 전역 시드 고정: {seed_val}")

    # 사용할 device 정의
    print("🚀 [2/8] 디바이스 설정 시작...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"---------- device : {device} ----------")
    print(f"PyTorch version: {torch.__version__}")
    print("✅ [2/8] 디바이스 설정 완료")

    # 사용할 모델과 tokenizer 불러오기
    print("🚀 [3/8] 모델 및 토크나이저 로드 시작...")
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    print("✅ [3/8] 모델 및 토크나이저 로드 완료")

    # 학습에 사용할 데이터셋 불러오기
    print("🚀 [4/8] 데이터셋 준비 시작...")
    preprocess_version = config['general'].get('preprocess_version', 'v1')
    processed_data_path = os.path.join(project_root, "data", "processed", preprocess_version)
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, processed_data_path, tokenizer)
    print("✅ [4/8] 데이터셋 준비 완료")

    # 학습 인수 설정
    print("🚀 [5/8] 학습 설정 시작...")
    training_args = setup_training_arguments(config)
    # setup_wandb(config, is_sweep)  # wandb 비활성화
    print("✅ [5/8] 학습 설정 완료")

    # Trainer 클래스 생성
    print("🚀 [6/8] 트레이너 설정 시작...")
    trainer = create_trainer(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset, training_args)
    print("✅ [6/8] 트레이너 설정 완료")
    
    # 모델 학습 시작
    print("🚀 [7/8] 모델 학습 시작...")
    print("➡️ 새로운 학습을 시작합니다.")
    results = trainer.train()
    print("✅ [7/8] 모델 학습 완료")

    # 최종 모델 저장 및 정리
    print("🚀 [8/8] 최종 정리 시작...")
    
    # 모델 저장
    final_model_path, model_dir_name = save_final_model(config, trainer, tokenizer)
    
    # 성능 평가 및 best 모델 업데이트
    update_best_model(config, final_model_path, model_dir_name, trainer)
    
    # WandB 비활성화됨
    
    print("✅ [8/8] 최종 정리 완료. 모든 프로세스가 성공적으로 끝났습니다.")
    
    # 최종 요약 (간결하게)
    print(f"\n🎉 학습 완료!")
    print(f"📁 저장: {os.path.relpath(final_model_path)}")
    print(f"🚀 평가: python src/models/evaluate.py --model_path outputs/models/best")
    print(f"🚀 추론: python src/models/infer.py --model_path outputs/models/best")
    
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART 모델 학습")
    parser.add_argument("--config-path", type=str, default=None, 
                       help="설정 파일 경로")
    parser.add_argument("--config-name", type=str, default="config.yaml",
                       help="설정 파일 이름 (sweep용)")
    
    # WandB Sweep에서 전달되는 하이퍼파라미터들
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="학습률")
    parser.add_argument("--num_train_epochs", type=int, default=None,
                       help="학습 에포크 수")
    parser.add_argument("--weight_decay", type=float, default=None,
                       help="가중치 감쇠")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                       help="배치 크기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="그래디언트 누적 스텝")
    parser.add_argument("--num_beams", type=int, default=None,
                       help="빔 서치 개수")
    parser.add_argument("--dropout", type=float, default=None,
                       help="드롭아웃 비율")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                       help="워밍업 비율")
    parser.add_argument("--label_smoothing", type=float, default=None,
                       help="라벨 스무딩 팩터")
    parser.add_argument("--length_penalty", type=float, default=None,
                       help="길이 페널티")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                       help="반복 페널티")
    
    args = parser.parse_args()
    
    # 명령줄 인자를 딕셔너리로 변환 (None이 아닌 값들만)
    sweep_overrides = {}
    if args.learning_rate is not None:
        sweep_overrides['learning_rate'] = args.learning_rate
    if args.num_train_epochs is not None:
        sweep_overrides['num_train_epochs'] = args.num_train_epochs
    if args.weight_decay is not None:
        sweep_overrides['weight_decay'] = args.weight_decay
    if args.per_device_train_batch_size is not None:
        sweep_overrides['per_device_train_batch_size'] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        sweep_overrides['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.num_beams is not None:
        sweep_overrides['num_beams'] = args.num_beams
    if args.dropout is not None:
        sweep_overrides['dropout'] = args.dropout
    if args.warmup_ratio is not None:
        sweep_overrides['warmup_ratio'] = args.warmup_ratio
    if args.label_smoothing is not None:
        sweep_overrides['label_smoothing'] = args.label_smoothing
    if args.length_penalty is not None:
        sweep_overrides['length_penalty'] = args.length_penalty
    if args.repetition_penalty is not None:
        sweep_overrides['repetition_penalty'] = args.repetition_penalty
    
    # WandB sweep 모드 비활성화됨
    # sweep 모드는 더 이상 지원되지 않음
    elif sweep_overrides:
        # 명령줄에서 sweep 파라미터가 전달된 경우
        print("🔄 명령줄 Sweep 파라미터로 실행")
        main(args.config_path, sweep_overrides)
    else:
        # 일반 실행
        print("🚀 일반 학습 모드로 실행")
        main(args.config_path)
