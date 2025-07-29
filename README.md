# NLP Summarization Competition

## Team

| ![문국현](https://avatars.githubusercontent.com/u/167870439?v=4) | ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![이승현](https://avatars.githubusercontent.com/u/126837633?v=4) | ![정재훈](https://avatars.githubusercontent.com/u/127591967?v=4) | ![조선미](https://avatars.githubusercontent.com/u/205017707?v=4) | ![이나경](https://avatars.githubusercontent.com/u/155069538?v=4) | ![이준석](https://avatars.githubusercontent.com/u/180180844?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [문국현](https://github.com/GH-Door) | [류지헌](https://github.com/mahomi) | [이승현](https://github.com/shyio06) | [정재훈](https://github.com/coevol) | [조선미](https://github.com/LearnSphere-2025) | [이나경](https://github.com/imnaagyeong) | [이준석](https://github.com/Lee-0624) |
| 팀장, 모델링 총괄 | 모델 실험 및 최적화 | EDA 및 데이터 전처리 | 모델 검증 및 성능 분석 | 모델 실험 및 최적화 | EDA 및 데이터 전처리 | 모델 검증 및 성능 분석 |

## 0. Overview
### Environment
- Python 3.10.13 이상
- CUDA 지원 GPU 환경 (PyTorch 2.7.1)
- uv 패키지 매니저

### Requirements

주요 라이브러리:
- `torch>=2.7.1` - 딥러닝 프레임워크
- `transformers>=4.54.0` - Hugging Face Transformers
- `pytorch-lightning>=2.5.2` - PyTorch Lightning
- `wandb>=0.21.0` - 실험 추적 및 시각화
- `accelerate>=0.26.0` - 분산 학습 지원
- `rouge>=1.0.1` - ROUGE 평가 지표
- `pandas>=2.3.1` - 데이터 처리
- `python-dotenv>=1.1.1` - 환경 변수 관리

## 1. Competiton Info

### Overview

- **대회명**: Dialogue Summarization | 일상 대화 요약
- **목표**: 일상 대화를 효과적으로 요약하는 모델 개발
- **주제**: 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 중 나누는 대화들에 대한 요약
- **설명**: 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 대화를 녹음해두더라도 전체를 다시 들을 수는 없기 때문에 요약이 필요하며, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

### Timeline

- **대회 시작**: 2025년 7월 25일 10:00
- **대회 종료**: 2025년 8월 6일 19:00

## 2. Components

### Directory

```
├── code
│   ├── baseline
│   │   ├── baseline.ipynb       # 베이스라인 Jupyter notebook 버전
│   │   ├── baseline.py          # 베이스라인 .py 버전
│   │   ├── config.yaml          # 모델 설정 파일
│   │   ├── solar_api.ipynb      # Solar API 활용 notebook
│   │   ├── solar_api.py         # Solar API 활용 .py 버전
│   └── utils
│       ├── check_gpu.py         # GPU 확인 유틸리티
│       └── log_util.py          # 로깅 유틸리티
├── input
│   ├── get_data.sh              # 데이터 다운로드 스크립트
│   └── data
│       ├── train.csv            # 학습 데이터
│       ├── dev.csv              # 검증 데이터
│       ├── test.csv             # 테스트 데이터
│       └── sample_submission.csv # 제출 예시
├── pyproject.toml               # 프로젝트 의존성 및 메타데이터 정의 (UV)
├── .python-version              # 프로젝트에서 사용할 Python 버전 지정 (UV)
├── uv.lock                      # 의존성 버전 잠금 파일 (UV)
└── README.md                    # 프로젝트 문서
```

### 참고 사항
- cd input && ./get_data.sh 를 실행하면 데이터를 다운로드 하여 data폴더가 생기게 됩니다.
- UV로 실행할 경우 다음 명령어로 실행할 수 있습니다:
```bash
uv run code/baseline/baseline.py
```

## 3. Data descrption

### Dataset overview

- **데이터셋 크기**:
  - Train: 12,457개
  - Dev: 499개
  - Test: 250개

- **데이터 구조**:
  - `fname`: 파일명 (예: train_0)
  - `dialogue`: 대화 내용 (최소 2명에서 최대 7명 참여)
  - `summary`: 대화 요약문 (train/dev에만 존재)
  - `topic`: 대화 주제

- **대화 참여자 표시**:
  - #Person1#, #Person2#, #Person3# 등으로 화자 구분
  - 개인정보는 #PhoneNumber#, #Address#, #PassportNumber# 등으로 마스킹

### EDA

- **대화 길이 분석**: 평균 512 토큰 이내의 대화
- **요약문 길이**: 평균 100 토큰 이내
- **주제 분포**: 일상생활, 직장, 학교, 의료, 쇼핑 등 다양한 주제
- **화자 수 분포**: 2-7명의 화자가 참여하는 대화

### Data Processing

- **Special Tokens 처리**:
  - 화자 토큰: #Person1#, #Person2#, #Person3#
  - 개인정보 마스킹: #PhoneNumber#, #Address#, #PassportNumber#
  - 토크나이저에 special_tokens로 추가하여 분해되지 않도록 처리

- **토크나이저 설정**:
  - Encoder 최대 길이: 512 토큰
  - Decoder 최대 길이: 100 토큰
  - BOS 토큰: `<s>`
  - EOS 토큰: `</s>`

- **데이터 전처리**:
  - Train 시: decoder_input에 BOS 토큰 추가, decoder_output에 EOS 토큰 추가
  - Inference 시: 생성된 요약문에서 특수 토큰 제거

## 4. Modeling

### Model descrition

- **사용 모델**: digit82/kobart-summarization
  - KoBART 기반 한국어 요약 전용 모델
  - Encoder-Decoder 구조의 Seq2Seq 모델
  - 한국어 뉴스 기사 요약으로 사전 학습됨

- **선택 이유**:
  - 한국어 요약 태스크에 특화된 사전 학습 모델
  - BART의 강력한 생성 능력과 한국어 이해 능력 결합
  - Fine-tuning을 통해 대화 요약 태스크에 적응 가능

### Modeling Process

#### 1. 학습 설정
- **Epochs**: 20
- **Learning Rate**: 1e-5
- **Batch Size**: 
  - Train: 50
  - Eval: 32
- **Optimizer**: AdamW
- **LR Scheduler**: Cosine
- **Mixed Precision**: FP16 사용

#### 2. 학습 전략
- **Early Stopping**: 
  - Patience: 3 epochs
  - Threshold: 0.001
- **평가 전략**: Epoch 단위 평가
- **모델 저장**: 최고 성능 모델 저장 (save_total_limit: 5)

#### 3. 생성 설정
- **Beam Search**: num_beams=4
- **No Repeat N-gram**: size=2
- **Max Length**: 100 토큰
- **Early Stopping**: True

#### 4. 학습 과정 모니터링
- Wandb를 통한 실시간 모니터링 (선택사항)
- ROUGE 점수 기반 성능 평가
- 학습/검증 손실 추적

## 5. Result

### Leader Board

- 평가 지표: ROUGE-L F1 Score
- 제출 파일 형식: CSV (fname, summary 컬럼)

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_

---

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_1)
