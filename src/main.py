import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import wandb

# 모듈 임포트
import src.utils as log
from src.data import Preprocess, prepare_train_dataset, prepare_test_dataset
from src.model import load_tokenizer_and_model_for_train, load_tokenizer_and_model_for_test
from src.train import load_trainer_for_train
from src.inference import inference
from src.check_gpu import get_device

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    # 디바이스 설정
    device = get_device()
    log.info(f"PyTorch version: {torch.__version__}")

    # 토크나이저 및 모델 로드 (학습용)
    generate_model_train, tokenizer_train = load_tokenizer_and_model_for_train(cfg)

    # 데이터 준비 (학습용)
    preprocessor_train = Preprocess(cfg.tokenizer.bos_token, cfg.tokenizer.eos_token)
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(cfg, preprocessor_train, tokenizer_train)

    # 트레이너 로드 및 학습
    trainer = load_trainer_for_train(cfg, generate_model_train, tokenizer_train, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    # 모델 학습이 완료된 후 wandb를 종료
    use_wandb = os.getenv('USE_WANDB', '').lower() == 'true'
    if cfg.training.report_to == 'wandb' and use_wandb:
        wandb.finish()

    log.info("Training complete. Starting inference...")

    # 추론 실행
    output_df = inference(cfg)
    log.info("Inference complete. Output saved to CSV.")
    log.info(output_df.head())

if __name__ == "__main__":
    main()
