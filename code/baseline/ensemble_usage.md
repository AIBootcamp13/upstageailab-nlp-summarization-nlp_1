# WandB Sweep 모델 앙상블 시스템 사용 가이드

## 1단계: 모델 학습 및 자동 저장

### WandB Sweep으로 여러 모델 학습
```bash
# 5개 모델을 학습하고 자동으로 ZIP 파일로 저장
python wandb_sweep.py --count 5
```

### 단일 모델 학습 (선택사항)
```bash
# 단일 모델 학습 (baseline.py에서도 자동 저장됨)
python baseline.py
```

## 2단계: 저장된 모델 확인

```bash
# models 폴더에 저장된 모델들 확인
ls models/
```

예상 출력:
```
model_baseline_20250804_123456.zip
model_baseline_20250804_124512.zip
model_baseline_20250804_125834.zip
model_baseline_20250804_131245.zip
```

## 3단계: 앙상블 추론 실행

### 모델 경로 설정
1. `ensemble_inference.py` 파일을 열기  
2. `run_single_method()` 함수와 `main()` 함수의 `model_paths` 리스트를 실제 저장된 모델 경로로 수정:

```python
model_paths = [
    "./models/model_baseline_20250804_123456.zip",  
    "./models/model_baseline_20250804_124512.zip",
    "./models/model_baseline_20250804_125834.zip",
    "./models/model_baseline_20250804_131245.zip"
]
```

### 실행 방법

#### 1. 모든 앙상블 방식 비교 (추천)
```bash
python ensemble_inference.py
# 또는
python ensemble_inference.py --mode=all
```

#### 2. 개별 앙상블 방식 실행
```bash
# 하드 보팅만 실행
python ensemble_inference.py --mode=hard_voting

# 소프트 보팅만 실행  
python ensemble_inference.py --mode=soft_voting

# 길이 기반만 실행
python ensemble_inference.py --mode=length_based

# 실시간 토큰 앙상블만 실행
python ensemble_inference.py --mode=realtime_token
```

#### 3. 도움말 확인
```bash
python ensemble_inference.py --help
```

## 4단계: 결과 확인

앙상블 실행 후 `ensemble_results/` 폴더에 다음 파일들이 생성됩니다:

```
ensemble_results/
├── ensemble_output_20250804_143022.csv              # 최종 앙상블 결과
├── individual_model_1_20250804_143022.csv           # 개별 모델 1 결과  
├── individual_model_2_20250804_143022.csv           # 개별 모델 2 결과
├── individual_model_3_20250804_143022.csv           # 개별 모델 3 결과
├── individual_model_4_20250804_143022.csv           # 개별 모델 4 결과
└── ensemble_metadata_20250804_143022.json           # 앙상블 메타데이터
```

## ZIP 파일 구조

각 모델 ZIP 파일은 다음과 같은 구조를 가집니다:

```
model_baseline_20250804_123456.zip
├── model.safetensors          # 모델 가중치
├── config.json               # 모델 설정
├── generation_config.json    # 생성 설정
├── tokenizer/               # 토크나이저 파일들
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.json
├── config.yaml              # 학습에 사용된 전체 설정
└── metadata.json            # WandB 정보 및 메타데이터
```

## 앙상블 전략

### 1. 하드 보팅 (Hard Voting)
- 각 모델이 독립적으로 완전한 텍스트 생성 후 토큰 단위 다수결
- 토큰별로 가장 많이 선택된 토큰을 최종 결과로 사용

### 2. 소프트 보팅 (Soft Voting) 
- 각 모델의 확률 분포를 평균하여 최적 후보 선택
- Beam search로 여러 후보 생성 후 평균 점수가 가장 높은 것 선택

### 3. 길이 기반 (Length-based)
- 각 모델의 생성 결과 중 가장 긴 텍스트를 선택
- 더 자세한 요약을 선호하는 방식

### 4. 실시간 토큰 앙상블 (Realtime Token Ensemble)
- 매 토큰 생성마다 모든 모델의 확률 분포를 평균
- 실시간으로 앙상블하여 가장 정교한 결과 생성

### 특징
- **동일 가중치**: 모든 모델이 동등하게 처리됨 (성능 기반 가중치 없음)
- **다양성 활용**: 서로 다른 하이퍼파라미터 조합의 장점을 모두 활용
- **성능 비교**: `--mode=all`로 모든 방식의 성능을 자동 비교 가능

## 문제 해결

### 모델 파일이 없는 경우
```
모델 파일 없음 (건너뜀): ./models/model_baseline_20250804_123456.zip
사용 가능한 모델 파일이 없습니다!
```

해결책: 먼저 WandB sweep을 실행하여 모델을 학습시키세요.

### GPU 메모리 부족
여러 모델을 동시에 로드하므로 GPU 메모리가 부족할 수 있습니다.
- 사용할 모델 수를 줄이기
- 배치 크기 줄이기 (코드 수정 필요)

### 토큰화 오류
서로 다른 토크나이저를 사용하는 모델들의 경우 토큰화 오류가 발생할 수 있습니다.
현재는 모든 모델이 동일한 베이스 모델(`digit82/kobart-summarization`)을 사용한다고 가정합니다.

## 성능 분석

앙상블 결과와 개별 모델 결과를 비교하여 앙상블의 효과를 분석할 수 있습니다:

1. ROUGE 스코어 계산
2. 개별 모델 대비 앙상블 성능 향상도 측정
3. 모델별 기여도 분석