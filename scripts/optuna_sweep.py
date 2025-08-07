#!/usr/bin/env python

import optuna
import subprocess
import json
import re
import os
import logging
import shutil
import time
import threading
from datetime import datetime


DISK_MONITOR_PATH = "/root/llm"  # 디스크 기준 경로
DISK_LIMIT_GB = 10  # 남은 공간이 10GB 미만이면 실행 중단


def get_available_disk_gb(path: str = DISK_MONITOR_PATH) -> float:
    """디스크 사용 가능 공간을 GB 단위로 반환"""
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def monitor_disk_usage(trial_number: int, stop_event: threading.Event) -> None:
    """디스크 사용 가능 공간 모니터링 - 부족 시 즉시 종료"""
    warning_gb = 12
    while not stop_event.is_set():
        free_gb = get_available_disk_gb()
        if free_gb < warning_gb:
            logging.warning(f"⚠️ Trial {trial_number} - 사용 가능 디스크 공간 {free_gb:.1f}GB (위험)")
        if free_gb < DISK_LIMIT_GB:
            logging.error(f"🚨 Trial {trial_number} - 사용 가능 디스크 공간 {free_gb:.1f}GB (한계 이하)")
            os._exit(1)
        time.sleep(30)


def objective(trial):
    """Optuna 목적 함수"""

    # 파라미터 제안 (고성능 모델 3개 기준으로 초정밀 탐색)
    # model_20250731_062917: Final Score 69.2073
    # model_20250731_113036: Final Score 69.2026  
    # model_20250731_120651: ROUGE-1 0.1587 (최신 고성능)
    # 모든 모델이 config.yaml 기본 설정으로 우수한 성능 달성
    learning_rate = trial.suggest_float('learning_rate', 8e-6, 1.2e-5, log=True)  # 1.0e-05 주변 더 넓은 탐색
    num_train_epochs = trial.suggest_categorical('num_train_epochs', [18, 19, 20, 21, 22])  # 20 epoch 중심 확장
    weight_decay = trial.suggest_float('weight_decay', 0.007, 0.013)  # 0.01 중심 확장된 정밀 탐색
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 24, 32])  # GPU 메모리 고려 축소
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.08, 0.12)  # 0.1 중심 확장된 정밀 탐색
    num_beams = trial.suggest_categorical('num_beams', [3, 4, 5, 6])  # 4 중심, 더 넓은 범위
    
    # train.py 지원 파라미터만 추가 (3번째 모델 패턴 반영)
    gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [2, 3, 4])  # 배치 크기 축소로 증가
    dropout = trial.suggest_float('dropout', 0.1, 0.2)  # 드롭아웃 범위 축소

    params = {
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'weight_decay': weight_decay,
        'per_device_train_batch_size': per_device_train_batch_size,
        'warmup_ratio': warmup_ratio,
        'num_beams': num_beams,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'dropout': dropout,
    }

    # 디스크 여유 공간 확인
    available_gb = get_available_disk_gb()
    if available_gb < DISK_LIMIT_GB:
        logging.error(f"Trial {trial.number} - 디스크 공간 부족 ({available_gb:.1f}GB)")
        raise optuna.TrialPruned()

    print(f"\n🧪 Trial {trial.number} 시작")
    print(f"📋 파라미터: {params}")
    print(f"💾 사용 가능 디스크: {available_gb:.1f}GB")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_disk_usage,
        args=(trial.number, stop_event),
        daemon=True
    )
    monitor_thread.start()

    cmd = ["python", "src/models/train.py", "--config-path", "src/config/config.yaml"]

    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=3600)
        if result.returncode == 0:
            logging.info(f"Trial {trial.number} - 학습 성공")

            eval_cmd = ["python", "src/models/evaluate.py", "--model_path", "outputs/models/latest"]
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=".", timeout=600)

            if eval_result.returncode == 0:
                match = re.search(r'Final Score: ([\d.]+)', eval_result.stdout)
                if match:
                    final_score = float(match.group(1))
                    print(f"✅ Trial {trial.number} 완료 - Final Score: {final_score}")
                    logging.info(f"Trial {trial.number} - Final Score: {final_score}")
                    return final_score
                else:
                    logging.error(f"Trial {trial.number} - Final Score 추출 실패")
                    raise optuna.TrialPruned()
            else:
                logging.error(f"Trial {trial.number} - 평가 실패")
                raise optuna.TrialPruned()
        else:
            logging.error(f"Trial {trial.number} - 학습 실패")
            raise optuna.TrialPruned()

    except subprocess.TimeoutExpired:
        logging.error(f"Trial {trial.number} - 타임아웃")
        raise optuna.TrialPruned()
    except Exception as e:
        print("📄 train.py stdout:\n", result.stdout)
        print("📄 train.py stderr:\n", result.stderr)
        logging.error(f"Trial {trial.number} - 예외 발생: {e}")
        raise optuna.TrialPruned()
    
    finally:
        stop_event.set()
        final_disk = get_available_disk_gb()
        logging.info(f"Trial {trial.number} 종료 - 최종 사용 가능 디스크: {final_disk:.1f}GB")

def run_optuna_study(n_trials=10):
    """Optuna 스터디 실행"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optuna_baseline_optimization.log'),
            logging.StreamHandler()
        ]
    )

    available_gb = get_available_disk_gb()
    logging.info(f"🚀 최적화 시작 - 사용 가능 디스크: {available_gb:.1f}GB")

    if available_gb < DISK_LIMIT_GB:
        logging.error(f"🚨 사용 가능 디스크 공간 부족: {available_gb:.1f}GB")
        return None

    study_name = f"baseline_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 고성능 모델 3개 패턴에 맞춘 고급 샘플러 설정
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,  # 더 많은 시작 trial로 안정성 확보
            n_ei_candidates=48,  # 더 많은 후보 탐색
            seed=42,
            multivariate=True,  # 파라미터간 상관관계 고려
            group=True  # 카테고리컬 파라미터 그룹화
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,  # 3개 모델 패턴 기반 조기 가지치기
            n_warmup_steps=2,
            interval_steps=1
        )
    )

    print(f"🚀 Optuna 스터디 시작: {study_name}")
    print(f"📊 총 {n_trials}번의 trial 예정")
    print(f"📋 config.yaml 기반 + 고성능 모델 3개 패턴 반영")
    print(f"🎯 목표: Final Score 69.2+ (최고 모델들 기준)")
    print(f"💾 디스크 여유 공간 제한: {DISK_LIMIT_GB}GB\n")

    study.optimize(objective, n_trials=n_trials)

    print("\n📊 최적화 결과")
    try:
        print(f"🏆 최고 점수: {study.best_value:.4f}")
        print(f"🎯 목표 달성 여부: {'✅ 달성!' if study.best_value >= 69.0 else '📈 개선 필요'}")
        print("🎛️ 최적 파라미터:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")
        
        # 상위 3개 trial 정보 표시
        if len(study.trials) >= 3:
            print("\n🥇 상위 3개 Trial:")
            sorted_trials = sorted([t for t in study.trials if t.value is not None], 
                                 key=lambda x: x.value, reverse=True)[:3]
            for i, trial in enumerate(sorted_trials, 1):
                print(f"  {i}. Trial {trial.number}: {trial.value:.4f}")
    except ValueError:
        print("⚠️ 성공한 trial이 없습니다.")

    results_file = f"optuna_baseline_results_{study_name}.json"
    try:
        with open(results_file, 'w') as f:
            # 상위 trials 정보 추가
            top_trials = []
            if study.trials:
                sorted_trials = sorted([t for t in study.trials if t.value is not None], 
                                     key=lambda x: x.value, reverse=True)[:5]
                top_trials = [{'trial_number': t.number, 'score': t.value, 'params': t.params} 
                            for t in sorted_trials]
            
            json.dump({
                'study_name': study_name,
                'reference_models': {
                    'model_20250731_062917': {'final_score': 69.2073, 'note': '최고 성능'},
                    'model_20250731_113036': {'final_score': 69.2026, 'note': '안정적 고성능'},
                    'model_20250731_120651': {'rouge_1': 0.1587, 'note': '최신 고성능 모델'}
                },
                'best_score': study.best_value if study.trials else None,
                'best_params': study.best_params if study.trials else None,
                'top_5_trials': top_trials,
                'config_base': 'config.yaml',
                'disk_limit_gb': DISK_LIMIT_GB,
                'total_trials': len(study.trials),
                'successful_trials': len([t for t in study.trials if t.value is not None])
            }, f, indent=2)
        print(f"💾 결과 저장: {results_file}")
    except Exception as e:
        logging.error(f"결과 저장 실패: {e}")

    return study


def main():
    """메인 함수"""
    print("🎯 Optuna Hyperparameter Optimization 시작")
    print("고성능 모델 3개 패턴 기반 초정밀 최적화")
    print("디스크 여유 공간 기준 최적화")

    total_gb = shutil.disk_usage(DISK_MONITOR_PATH).total / (1024**3)
    free_gb = get_available_disk_gb()
    used_gb = total_gb - free_gb

    print(f"\n💾 디스크 상태 ({DISK_MONITOR_PATH} 기준):")
    print(f"   - 전체: {total_gb:.1f}GB")
    print(f"   - 사용중: {used_gb:.1f}GB")
    print(f"   - 사용가능: {free_gb:.1f}GB")
    print(f"   - 제한: {DISK_LIMIT_GB}GB\n")

    if free_gb < DISK_LIMIT_GB:
        print("❌ 사용 가능 공간이 부족해 실행할 수 없습니다.")
        return

    study = run_optuna_study(n_trials=12)   # 고성능 모델 3개 기준 확장된 탐색
    if study:
        print("\n🎉 Optuna 최적화 완료!")
    else:
        print("\n❌ 디스크 부족으로 최적화 실패")


if __name__ == "__main__":
    main()


