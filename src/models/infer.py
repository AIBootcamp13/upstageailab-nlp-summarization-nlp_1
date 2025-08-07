"""
BART 기반 대화 요약 모델 추론 스크립트
train.py의 데이터 로딩 방식을 반영하여 리팩토링
"""

import pandas as pd
import os
import sys
import yaml
import torch
import argparse
from typing import Optional
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq, # 데이터 콜레이터 추가
)
from pathlib import Path
from datasets import load_dataset, Dataset # datasets 라이브러리 추가

# 현재 파일 위치 기준으로 data 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).parent.parent
sys.path.append(str(project_root / "src"))

from utils.postprocess import postprocess


# =========================================================================
# 추론용 데이터셋 준비 함수 (train.py 스타일로 리팩토링)
# =========================================================================

def prepare_test_dataset(config, tokenizer):
    test_file_path = os.path.join(project_root, config['test_file'])
    
    # HuggingFace datasets 라이브러리를 사용하여 CSV 로드
    raw_dataset = load_dataset('csv', data_files={'test': test_file_path})
    test_dataset = raw_dataset['test']
    
    print('-' * 150)
    print(f'test_data (from datasets):\n{test_dataset[0]["dialogue"]}')
    print('-' * 150)

    def tokenize_function(examples):
        # 모델 입력(인코더) 토크나이징 (레이블 없음)
        model_inputs = tokenizer(
            examples['dialogue'],
            max_length=config['tokenizer']['encoder_max_len'],
            truncation=True,
            padding='max_length'
        )
        return model_inputs

    # map 함수를 사용하여 데이터셋 전체에 토크나이징 적용
    tokenized_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
    
    print('-' * 10, 'Make dataset complete', '-' * 10)
    return test_dataset, tokenized_dataset # 원본과 토크나이징된 데이터셋 반환


# =========================================================================
# 모델 및 토크나이저 로딩 함수 (기존과 거의 동일, 일부 정리)
# =========================================================================

def load_tokenizer_and_model_for_test(config, device, model_path=None):
    print('-' * 10, 'Load tokenizer & model', '-' * 10)
    
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    if model_path:
        print('-' * 10, f'Using trained model: {model_path}', '-' * 10)
        generate_model = BartForConditionalGeneration.from_pretrained(model_path)
    else:
        print('-' * 10, f'Using base model: {model_name}', '-' * 10)
        generate_model = BartForConditionalGeneration.from_pretrained(model_name)
        
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)

    return generate_model, tokenizer


# =========================================================================
# 메인 추론 함수 (리팩토링)
# =========================================================================

def inference(config, model_path=None, sample_size=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'device : {device}', '-' * 10)
    print(f'PyTorch version: {torch.__version__}')

    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device, model_path)

    # 데이터셋 준비 (새로운 함수 호출)
    original_test_dataset, tokenized_test_dataset = prepare_test_dataset(config, tokenizer)
    
    # 샘플 크기 제한 (테스트용)
    if sample_size:
        print(f"🔧 {sample_size}개 샘플로 테스트합니다...")
        original_test_dataset = original_test_dataset.select(range(sample_size))
        tokenized_test_dataset = tokenized_test_dataset.select(range(sample_size))

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=generate_model)

    # DataLoader를 사용하여 배치 처리
    dataloader = torch.utils.data.DataLoader(
        tokenized_test_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        shuffle=False
    )
    
    summary = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="요약 생성"):
            # 필요한 데이터만 device로 이동 (값이 Tensor인 경우에만)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            gen_kwargs = {
                'max_length': config['inference'].get('generation_max_length', 128),
                'num_beams': config['inference']['num_beams'],
                'length_penalty': config['inference'].get('length_penalty', 1.0),
                'early_stopping': config['inference'].get('early_stopping', True),
            }
            # train.py의 generation_max_length와 일치시키거나, infer 설정 따로 관리
            
            generated_ids = generate_model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **gen_kwargs
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 후처리 적용
            postprocessed_preds = [postprocess(pred, config['inference']['remove_tokens']) for pred in decoded_preds]
            summary.extend(postprocessed_preds)

    # 결과물 DataFrame 생성
    output = pd.DataFrame({
        "fname": original_test_dataset['fname'],
        "summary": summary,
    })
    
    # 출력 디렉토리 생성
    output_file = config.get('output_file', 'outputs/predictions/submission.csv')
    # submission 파일일 경우 모델 정보(디렉토리명) 추가
    if os.path.basename(output_file).startswith('submission'):
        model_dir_name = os.path.basename(os.path.realpath(model_path)) if model_path else 'base'
        name, ext = os.path.splitext(output_file)
        output_file = f"{name}_{model_dir_name}{ext}"
    result_path = os.path.dirname(output_file)
    os.makedirs(result_path, exist_ok=True)
    
    # 리더보드 제출(sample_submission.csv) 형식: 첫 번째에 인덱스 컬럼이 포함되어 있음
    # submission*.csv 파일일 때만 index=True 로 저장하고, dev_pred 등은 기존처럼 index 제외
    include_index = os.path.basename(output_file).startswith("submission")
    output.to_csv(output_file, index=include_index)
    print(f"✅ 추론 완료! 결과가 저장되었습니다: {output_file}")
    print(f"📈 총 {len(summary)}개 예측 생성됨")

    # 최신 submission.csv 업데이트
    import shutil, os as _os
    if _os.path.basename(output_file).startswith("submission"):
        simple_sub = _os.path.join(_os.path.dirname(output_file), "submission.csv")
        shutil.copy(output_file, simple_sub)
        print(f"📋 {simple_sub} 업데이트 완료")

    return output


# =========================================================================
# 최고 성능 체크포인트 탐색 함수 (제거됨 - run_all.sh에서 처리)
# =========================================================================

def main(model_path, config_path=None, sample_size=None): # model_path를 필수 인자로 변경
    # Config 파일 경로 설정
    if config_path is None:
        config_path = os.path.join(project_root, "src", "config", "config_baseline.yaml")
    
    # Config 파일 로드
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # --- 최고 성능 모델 자동 검색 로직 (제거됨) ---
    if not model_path:
        raise ValueError("--model_path 인자는 필수입니다.")

    # 추론 관련 기본 설정 추가 (config.yaml에 명시하는 것이 더 좋음)
    if 'test_file' not in config:
        version = config['general'].get('preprocess_version', 'v1')
        config['test_file'] = f'data/processed/{version}/test.csv'
    if 'output_file' not in config:
        config['output_file'] = 'outputs/predictions/submission.csv'
    
    return inference(config, model_path, sample_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART 대화 요약 추론 스크립트")
    parser.add_argument("--model_path", help="모델 경로 (예: outputs/models/best, outputs/models/latest)")
    parser.add_argument("--config_path", default="src/config/config.yaml", help="Config YAML 경로")
    parser.add_argument("--sample_size", type=int, help="테스트용 샘플 개수(선택)")

    args = parser.parse_args()

    # model_path가 없으면 자동으로 best -> latest 순으로 선택
    def _auto_model_path(models_dir: str = "outputs/models") -> Optional[str]:
        best_link = os.path.join(models_dir, "best")
        if os.path.exists(best_link):
            return best_link
        latest_link = os.path.join(models_dir, "latest")
        if os.path.exists(latest_link):
            print("⚠️ 'best' 모델이 없어 'latest' 모델을 사용합니다.")
            return latest_link
        return None

    model_path = args.model_path
    if model_path is None:
        model_path = _auto_model_path()
        if model_path is None:
            print("❌ model_path가 지정되지 않았고 자동으로 사용할 모델을 찾을 수 없습니다.")
            sys.exit(1)
        print(f"🔍 자동 선택된 모델: {model_path}")

    main(model_path=model_path, config_path=args.config_path, sample_size=args.sample_size)
