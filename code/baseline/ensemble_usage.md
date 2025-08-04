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

### 방법 1: ensemble_inference.py 직접 수정
1. `ensemble_inference.py` 파일을 열기
2. `main()` 함수의 `model_paths` 리스트를 실제 저장된 모델 경로로 수정:

```python
model_paths = [
    "./models/model_baseline_20250804_123456.zip",  
    "./models/model_baseline_20250804_124512.zip",
    "./models/model_baseline_20250804_125834.zip",
    "./models/model_baseline_20250804_131245.zip"
]
```

3. 실행:
```bash
python ensemble_inference.py
```

### 방법 2: 모든 저장된 모델 자동 사용 (향후 개선 가능)
현재는 수동으로 경로를 지정해야 하지만, 필요시 `models/` 폴더의 모든 ZIP 파일을 자동으로 로드하도록 수정 가능합니다.

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

- **하드 보팅**: 각 모델이 독립적으로 완전한 텍스트 생성 후 토큰 단위 다수결
- **동일 가중치**: 모든 모델이 동등하게 처리됨 (성능 기반 가중치 없음)
- **다양성 활용**: 서로 다른 하이퍼파라미터 조합의 장점을 모두 활용

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