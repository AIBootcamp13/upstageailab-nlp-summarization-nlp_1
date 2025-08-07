# src/data/make_dataset.py 전처리 함수 기능 + 실행 코드

import os, sys, yaml
import pandas as pd
from pathlib import Path

# 현재 파일 위치 기준으로 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preprocessing_utils import clean_text, flatten_dialogue


def preprocess_file(input_path: str, output_path: str, is_test: bool = False, skip: bool=False):
    """skip=True 이면 파일을 그대로 복사만 한다"""
    if skip:
        import shutil
        shutil.copy(input_path, output_path)
        print(f"🚚 Copied raw -> {output_path}")
        return

    df = pd.read_csv(input_path)

    # dialogue: 정제 + 평탄화
    df["dialogue"] = df["dialogue"].apply(clean_text).apply(flatten_dialogue)

    # summary: 정제만 수행 (평탄화 X)
    if not is_test and "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_text)

    if is_test:
        df = df[["fname", "dialogue"]]

    df.to_csv(output_path, index=False)
    print(f"✅ Processed: {output_path}")


def main():
    # 프로젝트 루트 및 config 로드
    project_root = Path(__file__).resolve().parent.parent.parent
    with open(project_root / "src" / "config" / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    version = cfg.get("general", {}).get("preprocess_version", "v1")

    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / version
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("train.csv", False),
        ("dev.csv", False),
        ("test.csv", True),
    ]

    skip_flag = (version == 'v1')
    for fname, is_test in files:
        preprocess_file(input_dir / fname, output_dir / fname, is_test, skip=skip_flag)


if __name__ == "__main__":
    main()
