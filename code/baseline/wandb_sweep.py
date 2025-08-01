#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 사용방법
# 새 sweep 생성 후 5개 실험 실행
# python wandb_sweep.py --count 5
# 기존 sweep ID로 추가 실험 실행
# python wandb_sweep.py --sweep_id YOUR_SWEEP_ID --count 3


import os
import sys
import yaml
import wandb
import torch
from copy import deepcopy

# baseline.py에서 리팩토링된 함수들 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline import (
    load_config,
    train_model,
    load_tokenizer_and_model_for_train,
    prepare_train_dataset,
    load_trainer_for_train,
    Preprocess,
    compute_metrics,
    setup_wandb_login
)

def update_config_from_sweep(base_config, sweep_config):
    """
    WandB sweep에서 받은 하이퍼파라미터로 config 업데이트
    
    Args:
        base_config: 기본 설정
        sweep_config: wandb sweep에서 전달된 설정
    
    Returns:
        업데이트된 설정
    """
    # 기본 설정을 복사하여 수정
    config = deepcopy(base_config)
    
    # wandb sweep 파라미터를 config에 반영
    for key, value in sweep_config.items():
        # key가 'training.learning_rate' 형태인 경우 처리
        if '.' in key:
            section, param = key.split('.', 1)
            if section in config and param in config[section]:
                config[section][param] = value
                print(f"Updated {section}.{param} = {value}")
    
    return config

def compute_metrics_with_avg(config, tokenizer, pred):
    """
    ROUGE 점수를 계산하고 평균값을 추가하는 함수
    """
    # 기본 ROUGE 점수 계산
    result = compute_metrics(config, tokenizer, pred)
    
    # ROUGE-1, ROUGE-2, ROUGE-L의 평균 계산
    rouge_avg = (result.get('rouge-1', 0) + result.get('rouge-2', 0) + result.get('rouge-l', 0)) / 3
    result['rouge_avg'] = rouge_avg
    
    return result

def train_sweep():
    """
    WandB sweep 실행을 위한 훈련 함수
    """
    # wandb 초기화
    run = wandb.init()
    
    # sweep 설정 가져오기
    sweep_config = wandb.config
    
    # 기본 config 로드
    base_config = load_config()
    
    # 기본 config를 sweep 파라미터로 업데이트
    config = update_config_from_sweep(base_config, sweep_config)
    
    # sweep에서는 항상 wandb를 사용하므로 로그인 처리
    if not setup_wandb_login():
        raise ValueError("wandb sweep을 사용하려면 WANDB_API_KEY가 필요합니다. .env 파일을 확인하세요.")
    
    # wandb 사용 설정
    config['training']['report_to'] = 'wandb'
    
    # 메인 훈련 함수 실행
    try:
        # 사용할 device를 정의합니다.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 사용할 모델과 tokenizer를 불러옵니다.
        generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
        
        # 학습에 사용할 데이터셋을 불러옵니다.
        preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
        data_path = config['general']['data_path']
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)
        
        # Trainer 클래스를 불러옵니다. (compute_metrics_with_avg 사용)
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
        from baseline import LoggingCallback
        
        # set training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=config['general']['output_dir'],
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
            eval_strategy=config['training']['evaluation_strategy'],
            save_strategy=config['training']['save_strategy'],
            save_total_limit=config['training']['save_total_limit'],
            fp16=config['training']['fp16'],
            load_best_model_at_end=config['training']['load_best_model_at_end'],
            seed=config['training']['seed'],
            logging_dir=config['training']['logging_dir'],
            logging_strategy=config['training']['logging_strategy'],
            predict_with_generate=config['training']['predict_with_generate'],
            generation_max_length=config['training']['generation_max_length'],
            do_train=config['training']['do_train'],
            do_eval=config['training']['do_eval'],
            report_to=config['training']['report_to']
        )
        
        # wandb 초기화 (이미 위에서 했지만 확실히 하기 위해)
        if config['training']['report_to'] == 'wandb':
            # wandb 설정
            os.environ["WANDB_LOG_MODEL"] = "true"
            os.environ["WANDB_WATCH"] = "false"
        
        # EarlyStopping 콜백
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping_patience'],
            early_stopping_threshold=config['training']['early_stopping_threshold']
        )
        
        # Trainer 클래스를 정의합니다.
        trainer = Seq2SeqTrainer(
            model=generate_model,
            args=training_args,
            train_dataset=train_inputs_dataset,
            eval_dataset=val_inputs_dataset,
            compute_metrics=lambda pred: compute_metrics_with_avg(config, tokenizer, pred),
            callbacks=[early_stopping_callback, LoggingCallback()]
        )
        
        # 모델 학습을 시작합니다.
        trainer.train()
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.finish(exit_code=1)
        raise
    
    # 정상 종료
    wandb.finish()

def create_sweep_from_yaml(yaml_path="config_sweep.yaml", project_name=None):
    """
    YAML 파일에서 sweep 설정을 읽어와 sweep 생성
    
    Args:
        yaml_path: sweep 설정 YAML 파일 경로
        project_name: wandb 프로젝트 이름 (None이면 config에서 읽음)
    
    Returns:
        sweep_id
    """
    # sweep 설정 읽기
    with open(yaml_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # 프로젝트 이름 설정
    if project_name is None:
        base_config = load_config()
        project_name = base_config.get('wandb', {}).get('project', 'dialogue-summarization-sweep')
    
    # sweep 생성
    sweep_id = wandb.sweep(
        sweep_config,
        project=project_name
    )
    
    return sweep_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WandB Sweep for Dialogue Summarization')
    parser.add_argument('--sweep_id', type=str, help='기존 sweep ID (없으면 새로 생성)')
    parser.add_argument('--sweep_config', type=str, default='config_sweep.yaml', 
                        help='Sweep 설정 파일 경로')
    parser.add_argument('--count', type=int, default=1, 
                        help='실행할 sweep 실험 수')
    parser.add_argument('--project', type=str, default=None,
                        help='WandB 프로젝트 이름')
    
    args = parser.parse_args()
    
    # sweep ID가 없으면 새로 생성
    if args.sweep_id is None:
        print(f"Creating new sweep from {args.sweep_config}...")
        sweep_id = create_sweep_from_yaml(args.sweep_config, args.project)
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Using existing sweep ID: {sweep_id}")
    
    # sweep agent 실행
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=args.count
    )
    
    print(f"Completed {args.count} sweep runs!")