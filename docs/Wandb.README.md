# WandB 실험 관리 가이드

팀 실험 관리 및 하이퍼파라미터/모델 버전 기록을 위한 WandB 설정 방법입니다.

---

## 1. 계정 생성 및 API 키 등록
1. https://wandb.ai 에서 개인 계정 가입 (또는 GitHub 계정 연동)
2. WandB 대시보드에서 API Key를 발급
3. 서버 내부 터미널에서 `conda activate Project Name` 후 가상환경 진입 or UV
4. `pip install wandb` > `wandb login [발급받은_API_KEY]`


## 2. 팀 & 프로젝트 연결
- Organization (entity): moonstalker9010-none
- Project: NLP

실험 코드에서 아래와 같이 프로젝트에 연결

```python
import wandb

wandb.init(
    project="NLP",
    entity="moonstalker9010-none",
    name="baseline-experiment", # Wandb Run 실험관리 이름
    config={ # 하이퍼파라미터 값을 지정할 떄 같은 이름으로 지정해야 실제 Run 화면에서 중복된 파라미터가 안 나옵니다~!
        # 예를 들어 "learning_rate" 라고 했는데 다른 분이 LR로 하면 같은 러닝레이트인데 파라미터값이 2개가 생성되기 떄문에
        # 혼동을 유발할 수 있을 거 같아 말씀드립니다. 이건 따로 추후에 공통으로 지정하던지 아니면 먼저 하신분의 파라미터 값 이름을 그대로
        # 이어받아서 해야할 거 같아요  
        "learning_rate": 0.001, # 예시
        "batch_size": 32,
        "epochs": 10,
    }
)
```


## 3. WandB 실험 종료

모델 학습이 끝나면 반드시 `wandb.finish()`를 호출하여 실험을 종료.  
이 과정이 누락되면 실험 기록이 WandB 서버에 제대로 저장되지 않을 수 있음  

```python
wandb.finish()
```


## 4. 모델 파일(아티팩트) 업로드

학습한 모델을 WandB에 아티팩트로 업로드하면 실험별 모델 파일 관리가 가능

```python
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model_best.pth')  # 저장한 모델 파일명에 맞게 수정
wandb.log_artifact(artifact)
```


## 5. 대시보드 확인

- WandB 대시보드 (https://wandb.ai/)에서 Organization > Project > Runs 메뉴로 이동
- 각 실험별로 로그된 하이퍼파라미터, 학습 곡선, 모델 아티팩트 확인 가능
- 여러 Run을 선택해 그래프를 동시에 비교할 수 있음


## 6. 참고 명령어

- 로그인 재설정  
  ```bash
  wandb login --relogin
  ```

- WandB 로컬 캐시 삭제  
  ```bash
  wandb sync --clean
  ```


팀원 모두 위 가이드를 참고하여 동일한 WandB Organization과 Project로 실험을 기록하고, 대시보드에서 진행 상황을 공유해주세요.