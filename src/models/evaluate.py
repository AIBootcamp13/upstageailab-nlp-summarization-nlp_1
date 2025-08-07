#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""evaluate.py · Dialog Summarization 경진대회 평가 스크립트"""

import os
import sys
import argparse
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Optional

# 현재 파일 위치 기준으로 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# ✅ 중앙화된 metric 유틸리티 및 후처리 함수 import
from src.utils.metrics import calculate_rouge_scores
from src.utils.postprocess import postprocess

def find_best_model_path(models_dir: str = "outputs/models") -> Optional[str]:
    """최고 성능 모델 경로를 자동으로 찾기"""
    best_link = os.path.join(models_dir, "best")
    if os.path.exists(best_link):
        return best_link
    
    # best 링크가 없으면 latest 시도
    latest_link = os.path.join(models_dir, "latest")
    if os.path.exists(latest_link):
        print("⚠️ 'best' 모델을 찾을 수 없어 'latest' 모델을 사용합니다.")
        return latest_link
    
    return None


def generate_predictions(model_path: str, ref_file: str, output_file: str) -> str:
    """모델로 예측 생성"""
    print(f"🚀 모델 로드: {model_path}")
    
    try:
        from src.models.infer import inference as _infer
        
        # config 로드 및 설정
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "src" / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["test_file"] = ref_file

        # test_pred 파일명에 모델 정보 추가
        if os.path.basename(output_file).startswith("test_pred"):
            model_dir_name = os.path.basename(os.path.realpath(model_path)) if model_path else "model"
            name, ext = os.path.splitext(output_file)
            output_file = f"{name}_{model_dir_name}{ext}"
        config["output_file"] = output_file

        print("📝 예측 생성 중...")
        _infer(config, model_path=model_path)
        print(f"✅ 예측 완료: {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"❌ 예측 생성 실패: {e}")
        raise


def evaluate_predictions(pred_csv: str, ref_csv: str) -> Dict[str, float]:
    """예측 결과 평가"""
    print(f"📊 평가 시작...")
    
    # 파일 검증
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"예측 파일을 찾을 수 없습니다: {pred_csv}")
    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"정답 파일을 찾을 수 없습니다: {ref_csv}")
    
    # 데이터 로드
    pred_df = pd.read_csv(pred_csv)
    ref_df = pd.read_csv(ref_csv)
    
    # 컬럼 검증
    if not {"fname", "summary"}.issubset(pred_df.columns):
        raise ValueError("예측 파일에 'fname', 'summary' 컬럼이 없습니다.")
    if not {"fname", "summary"}.issubset(ref_df.columns):
        raise ValueError("정답 파일에 'fname', 'summary' 컬럼이 없습니다.")

    # 데이터 병합
    merged = pd.merge(
        ref_df[["fname", "summary"]],
        pred_df[["fname", "summary"]].rename(columns={"summary": "pred"}),
        on="fname", 
        how="inner"
    )
    
    if len(merged) == 0:
        raise ValueError("예측과 정답 파일의 fname이 일치하지 않습니다.")
    
    print(f"📈 평가 대상: {len(merged)}개 샘플")

    # 예측값과 정답 추출
    preds = merged["pred"].tolist()
    refs_list = [str(s).split('|||' ) for s in merged["summary"]]

    # 후처리: train.py 와 동일한 규칙 적용
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = project_root / "src" / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        _cfg = yaml.safe_load(f)
    remove_tokens = _cfg.get('inference', {}).get('remove_tokens', None)
    preds = [postprocess(p, remove_tokens) for p in preds]

    # ROUGE 점수 계산
    scores = calculate_rouge_scores(preds, refs_list)
    
    return scores


def print_results(scores: Dict[str, float], model_info: str = ""):
    """결과 출력"""
    print(f"\n{'='*50}")
    print(f"📊 평가 결과 {model_info}")
    print(f"{'='*50}")
    print(f"ROUGE-1: {scores['rouge_1']:.4f}")
    print(f"ROUGE-2: {scores['rouge_2']:.4f}")
    print(f"ROUGE-L: {scores['rouge_l']:.4f}")
    print(f"{'='*50}")
    print(f"🎯 Final Score: {scores['final_score']:.4f}")
    print(f"{'='*50}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Dialog Summarization 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 최고 성능 모델 평가 (권장)
  python evaluate.py --model_path outputs/models/best
  
  # 특정 모델 평가
  python evaluate.py --model_path outputs/models/model_20250727_145129
  
  # 기존 예측 파일 평가
  python evaluate.py --pred_file outputs/predictions/test_pred.csv
  
  # 자동 모델 찾기 (best 모델 자동 선택)
  python evaluate.py --auto
        """
    )

    # 하나의 옵션이라도 지정되지 않으면 --auto 가 기본값으로 동작하도록 설정
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--model_path", 
                       help="모델 경로 지정 (예: outputs/models/best, outputs/models/latest)")
    group.add_argument("--pred_file", 
                       help="기존 예측 CSV 파일로 평가 (모델 로드 없이 바로 평가)")
    group.add_argument("--auto", action="store_true", 
                       help="자동으로 최고 성능 모델 선택 (가장 간편한 방법)")

    parser.add_argument("--ref_file", default=None, 
                       help="정답 CSV 파일 경로 (기본값: data/processed/test.csv)")
    parser.add_argument("--output_file", default="outputs/predictions/test_pred.csv",
                       help="예측 결과 저장 경로 (기본값: outputs/predictions/test_pred.csv)")

    args = parser.parse_args()

    # 인수가 아무것도 없으면 자동 모드로 전환
    if not (args.auto or args.model_path or args.pred_file):
        args.auto = True
        print("⚙️  인수가 없어서 자동 모드(--auto)로 실행합니다.")

    # ref_file 기본값 (가장 먼저 처리)
    if args.ref_file is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cfg_path = project_root / "src" / "config" / "config.yaml"
        with open(cfg_path, "r") as f:
            _cfg = yaml.safe_load(f)
        version = _cfg.get('general', {}).get('preprocess_version', 'v1')
        args.ref_file = f"data/processed/{version}/dev.csv"

    try:
        # 1. 모델 경로 결정
        if args.auto:
            model_path = find_best_model_path()
            if model_path is None:
                print("❌ 사용 가능한 모델을 찾을 수 없습니다.")
                print("💡 먼저 모델을 학습하거나 정확한 경로를 지정하세요.")
                return
            print(f"🔍 자동 선택된 모델: {model_path}")
            pred_path = generate_predictions(model_path, args.ref_file, args.output_file)
            model_info = f"({os.path.basename(model_path)} 모델)"
            
        elif args.model_path:
            if not os.path.exists(args.model_path):
                print(f"❌ 모델 경로를 찾을 수 없습니다: {args.model_path}")
                return
            pred_path = generate_predictions(args.model_path, args.ref_file, args.output_file)
            model_info = f"({os.path.basename(args.model_path)} 모델)"
            
        else:  # args.pred_file
            pred_path = args.pred_file
            model_info = "(기존 예측 파일)"

        # 2. 평가 수행
        scores = evaluate_predictions(pred_path, args.ref_file)
        
        # 3. 결과 출력
        print_results(scores, model_info)

        # 최신 test_pred.csv 심플 파일 업데이트
        import shutil, os as _os
        if _os.path.basename(pred_path).startswith("test_pred"):
            simple_path = _os.path.join(_os.path.dirname(pred_path), "test_pred.csv")
            shutil.copy(pred_path, simple_path)
            print(f"📋 {simple_path} 업데이트 완료")
         
        return scores

    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        return None


if __name__ == "__main__":
    main()
