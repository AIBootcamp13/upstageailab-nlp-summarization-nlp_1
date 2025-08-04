#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 사용방법
# 새 sweep 생성 후 5개 실험 실행
# python wandb_sweep.py --count 5

# 기존 sweep ID로 추가 실험 실행
# python wandb_sweep.py --sweep_id YOUR_SWEEP_ID --count 3

# 파라미터 오버라이드 (특정 값 고정)
# python wandb_sweep.py --count 1 --override training.num_train_epochs=1

# 여러 파라미터 동시 고정
# python wandb_sweep.py --count 5 \
#     --override training.learning_rate=5e-5 \
#     --override training.num_train_epochs=10 \
#     --override training.warmup_ratio=0.1
import sys
import yaml
import wandb
from copy import deepcopy
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# baseline.py에서 리팩토링된 함수들 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline import (
    load_config,
    compute_metrics,
    setup_wandb_login,
    inference,
    train_model
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
    ROUGE 점수를 계산하고 평균값을 추가하는 함수 (WandB sweep용)
    """
    # 기본 ROUGE 점수 계산
    result = compute_metrics(config, tokenizer, pred)
    
    # ROUGE-1, ROUGE-2, ROUGE-L의 평균 계산
    rouge_avg = (result.get('rouge-1', 0) + result.get('rouge-2', 0) + result.get('rouge-l', 0)) / 3
    result['rouge_avg'] = rouge_avg
    
    return result

def convert_parameter_value(value_str):
    """
    문자열 값을 적절한 타입으로 변환하는 함수
    
    Args:
        value_str: 변환할 문자열 값
        
    Returns:
        변환된 값 (int, float, bool, str)
    """
    # boolean 값 처리
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # 숫자 값 처리
    try:
        # 정수인지 확인
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
        # 실수로 처리
        return float(value_str)
    except ValueError:
        # 문자열로 처리
        return value_str

def apply_parameter_overrides(sweep_config, overrides):
    """
    sweep 설정에 파라미터 오버라이드를 적용하는 함수
    
    Args:
        sweep_config: WandB sweep 설정 딕셔너리
        overrides: 오버라이드할 파라미터 리스트 (예: ['training.learning_rate=1e-4'])
    """
    if not overrides:
        return
    
    print(f"파라미터 오버라이드 적용: {len(overrides)}개")
    
    for override in overrides:
        if '=' not in override:
            print(f"잘못된 오버라이드 형식 (건너뜀): {override}")
            continue
            
        key, value_str = override.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # 값 타입 변환
        converted_value = convert_parameter_value(value_str)
        
        # sweep 설정에서 해당 파라미터를 고정값으로 변경
        if key in sweep_config.get('parameters', {}):
            old_config = sweep_config['parameters'][key]
            sweep_config['parameters'][key] = {'value': converted_value}
            print(f"  {key}: {old_config} → 고정값: {converted_value}")
        else:
            print(f"  {key}: 새 파라미터로 추가: {converted_value}")
            sweep_config.setdefault('parameters', {})[key] = {'value': converted_value}

def train_sweep():
    """
    WandB sweep 실행을 위한 훈련 함수 (baseline.py의 train_model 사용)
    """
    # sweep에서는 항상 wandb를 사용하므로 먼저 로그인 처리
    if not setup_wandb_login():
        raise ValueError("wandb sweep을 사용하려면 WANDB_API_KEY가 필요합니다. .env 파일을 확인하세요.")
    
    # wandb 초기화 (로그인 후)
    wandb.init()
    
    # sweep 설정 가져오기
    sweep_config = wandb.config
    
    # 기본 config 로드
    base_config = load_config()
    
    # 기본 config를 sweep 파라미터로 업데이트
    config = update_config_from_sweep(base_config, sweep_config)
    
    # 업데이트된 config 출력
    print("업데이트된 config 파라미터:")
    print(f"  generation_max_length: {config['training']['generation_max_length']}")
    print(f"  encoder_max_len: {config['tokenizer']['encoder_max_len']}")
    print(f"  decoder_max_len: {config['tokenizer']['decoder_max_len']}")
    print(f"  learning_rate: {config['training']['learning_rate']}")
    print(f"  seed: {config['training']['seed']}")
    
    # 재현성을 위한 시드 설정
    from baseline import set_seed_for_reproducibility
    final_seed = config['training']['seed']  # sweep에서 업데이트된 시드 또는 기본값 42
    set_seed_for_reproducibility(final_seed)
    print(f"실험 시드 설정: {final_seed}")
    
    # wandb 사용 설정
    config['training']['report_to'] = 'wandb'
    
    # compute_metrics를 sweep용 함수로 교체 (원복 불필요)
    import baseline
    baseline.compute_metrics = compute_metrics_with_avg
    
    try:
        # baseline.py의 train_model 함수 사용 (자동으로 모델 저장됨)
        model, tokenizer = train_model(config)
        print("baseline.py의 train_model 함수를 통한 학습 및 모델 저장 완료")
        
        # 학습 완료 후 테스트 데이터로 추론 실행
        try:
            print("학습 완료. 테스트 데이터로 추론을 시작합니다...")
            
            # 추론 실행 (학습된 최상의 모델과 토크나이저 사용)
            # inference() 함수 내부에서 자동으로 WandB 아티팩트 업로드됨
            inference(config, model=model, tokenizer=tokenizer)
                
        except Exception as inference_error:
            print(f"추론 중 오류가 발생했습니다: {inference_error}")
            # 추론 실패해도 학습은 성공했으므로 계속 진행
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.finish(exit_code=1)
        raise
    
    # 정상 종료
    wandb.finish()

def create_sweep_from_yaml(yaml_path="config_sweep.yaml", project_name=None, overrides=None):
    """
    YAML 파일에서 sweep 설정을 읽어와 sweep 생성
    
    Args:
        yaml_path: sweep 설정 YAML 파일 경로
        project_name: wandb 프로젝트 이름 (None이면 config에서 읽음)
        overrides: 오버라이드할 파라미터 리스트
    
    Returns:
        sweep_id
    """
    # sweep 설정 읽기
    with open(yaml_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # 파라미터 오버라이드 적용
    if overrides:
        apply_parameter_overrides(sweep_config, overrides)
    
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
    parser.add_argument('--override', action='append', default=[],
                        help='파라미터 오버라이드 (예: training.learning_rate=1e-4)')
    
    args = parser.parse_args()
    
    # 오버라이드 파라미터 출력
    if args.override:
        print(f"파라미터 오버라이드: {args.override}")
    
    # sweep ID 결정 (우선순위: CLI 인자 > 환경변수 > 새로 생성)
    if args.sweep_id is not None:
        sweep_id = args.sweep_id
        print(f"Using sweep ID from command line: {sweep_id}")
    else:
        # 환경변수에서 sweep ID 확인
        env_sweep_id = os.getenv('WANDB_SWEEP_ID')
        if env_sweep_id and env_sweep_id.strip():
            sweep_id = env_sweep_id.strip()
            print(f"Using sweep ID from environment variable: {sweep_id}")
        else:
            # 새로운 sweep 생성
            print(f"Creating new sweep from {args.sweep_config}...")
            
            # WandB가 빈 WANDB_SWEEP_ID 환경변수를 감지하지 않도록 임시 제거
            temp_sweep_id = os.environ.pop('WANDB_SWEEP_ID', None)
            
            try:
                sweep_id = create_sweep_from_yaml(args.sweep_config, args.project, args.override)
                print(f"Created sweep with ID: {sweep_id}")
                print(f"To resume this sweep later, add 'WANDB_SWEEP_ID={sweep_id}' to your .env file")
            finally:
                # 원래 환경변수 복원 (있었다면)
                if temp_sweep_id is not None:
                    os.environ['WANDB_SWEEP_ID'] = temp_sweep_id
    
    # sweep agent 실행
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=args.count
    )
    
    print(f"Completed {args.count} sweep runs!")