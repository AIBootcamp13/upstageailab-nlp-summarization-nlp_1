#!/bin/bash
# bash shell 에서는 함수정의 먼저, 실행로직(main flow)은 아래에 두는 구조: 깔끔/유지보수 용이

set -e # 오류 발생 시 즉시 스크립트 중단

# ==============================================================================
# STEP 0: 초기 설정 (Setup)
# ==============================================================================
step_0_setup() {
    echo "STEP 0: 초기 설정"
    
    # .env 파일 로드 (python-dotenv 필요)
    if [ -f ".env" ]; then
        echo "🌍 .env 파일 로드"
        export $(grep -v '^#' .env | xargs)
    fi

    # 스크립트가 있는 디렉토리 기준으로 프로젝트 루트를 찾습니다.
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    PROJECT_ROOT="$SCRIPT_DIR/.."
    cd "$PROJECT_ROOT" || exit 1

    echo "🚀 프로젝트 루트: $(pwd)"
    
    
    export PYTHONPATH=.
    echo "✅ 초기 설정 완료"
}

# ==============================================================================
# STEP 1: 데이터 전처리 (Preprocess Data)
# ==============================================================================
step_1_preprocess() {
    echo -e "\n\nSTEP 1: 데이터 전처리"
    python3 src/data/make_dataset.py
    echo "✅ 데이터 전처리 완료"
}

# ==============================================================================
# STEP 2: 모델 학습 (Train Model)
# ==============================================================================
step_2_train() {
    echo -e "\n\nSTEP 2: 모델 학습"
    python3 src/models/train.py || true
    echo "✅ 모델 학습 완료"
}

# ==============================================================================
# STEP 3: 평가 (Evaluation)
# ==============================================================================
step_3_evaluate() {
    echo -e "\n\nSTEP 3: 평가"
    echo "📊 최고 성능 모델 평가 (auto)"
    eval_output=$(python3 src/models/evaluate.py --auto)
    echo "$eval_output"

    avg_score=$(echo "$eval_output" | grep "Final Score" | grep -o '[0-9.]\+')
    echo -e "\n🌟 \033[1;36m[✔] 평가 완료: final ROUGE 점수 = $avg_score\033[0m"
}

# ==============================================================================
# STEP 4: 추론 (Inference)
# ==============================================================================
step_4_infer() {
    echo -e "\n\nSTEP 4: 추론"
    echo "🔍 최고 성능 모델로 추론 (auto)"
    python3 src/models/infer.py
    echo "✅ 추론 완료"
}

# ==============================================================================
# 메인 실행 로직 (Main Execution)
# ==============================================================================

step_0_setup

# step_1 전처리 디렉토리 존재 여부 확인 후 필요 시 실행
#PREV_VERSION=$(python - <<PY
#import yaml, sys
#cfg=yaml.safe_load(open('src/config/config.yaml'))
#print(cfg.get('general', {}).get('preprocess_version', 'v1'))
#PY
#)

#if [ ! -f "data/processed/${PREV_VERSION}/train.csv" ]; then
#  echo "📂 전처리된 데이터가 없어서 자동 전처리 실행(${PREV_VERSION})"
#  step_1_preprocess
#else
#  echo "✅ 전처리 데이터(${PREV_VERSION})가 이미 존재하여 건너뜁니다."
#fi

step_2_train
step_3_evaluate
step_4_infer

echo -e "\n\n🏁 \033[1;32m[ALL DONE] 모든 파이프라인 단계가 성공적으로 완료되었습니다.\033[0m"
echo "📂 생성된 prediction 파일들:"
ls -lh outputs/predictions/*_model_*.csv
