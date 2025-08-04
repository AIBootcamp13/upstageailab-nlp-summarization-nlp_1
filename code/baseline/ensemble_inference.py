#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Inference System for Hard Voting
WandB sweep으로 생성된 여러 모델들을 하드 보팅으로 앙상블하여 추론하는 시스템
"""

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append('../utils')
import log_util as log

import pandas as pd
import json
import yaml
import torch
import zipfile
import shutil
import time
from datetime import datetime
from collections import Counter
from tqdm import tqdm

from transformers import AutoTokenizer, BartForConditionalGeneration

# baseline.py에서 필요한 클래스들 임포트
from baseline import Preprocess

def load_model_package(zip_path):
    """
    ZIP 파일에서 모델, 토크나이저, 설정을 로딩
    
    Args:
        zip_path: ZIP 파일 경로
        
    Returns:
        tuple: (model, tokenizer, config, metadata)
    """
    temp_dir = f"temp_load_{int(time.time())}"
    
    try:
        log.info(f"모델 패키지 로딩 시작: {zip_path}")
        
        # ZIP 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 설정 로드
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info("설정 파일 로드 완료")
            
        # 메타데이터 로드
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        log.info("메타데이터 로드 완료")
        
        # 토크나이저 로드
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        log.info("토크나이저 로드 완료")
        
        # 모델 로드
        model = BartForConditionalGeneration.from_pretrained(temp_dir)
        model.resize_token_embeddings(len(tokenizer))
        log.info("모델 로드 완료")
        
        return model, tokenizer, config, metadata
        
    except Exception as e:
        log.error(f"모델 패키지 로딩 중 오류: {e}")
        raise
        
    finally:
        # 임시 폴더 삭제
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

class HardVotingEnsemble:
    """
    하드 보팅 기반 앙상블 추론 클래스
    각 모델이 독립적으로 완전한 텍스트를 생성한 후 토큰 단위로 다수결
    """
    
    def __init__(self, model_paths, device="cuda:0"):
        """
        Args:
            model_paths: 모델 ZIP 파일 경로들의 리스트
            device: 사용할 디바이스
        """
        self.model_paths = model_paths
        self.device = device
        self.models = []
        self.tokenizers = []
        self.configs = []
        self.metadata_list = []
        
        log.info(f"앙상블 시스템 초기화: {len(model_paths)}개 모델")
        log.info(f"사용 디바이스: {device}")
    
    def load_models(self):
        """모든 모델들을 로딩"""
        log.info("모델들 로딩 시작...")
        
        for i, path in enumerate(self.model_paths):
            log.info(f"모델 {i+1}/{len(self.model_paths)} 로딩 중: {path}")
            
            try:
                model, tokenizer, config, metadata = load_model_package(path)
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.configs.append(config)
                self.metadata_list.append(metadata)
                
                log.info(f"모델 {i+1} 로딩 완료: {metadata.get('wandb_run_name', 'Unknown')}")
                
            except Exception as e:
                log.error(f"모델 {i+1} 로딩 실패: {e}")
                raise
        
        log.info(f"총 {len(self.models)}개 모델 로딩 완료")
    
    def generate_with_single_model(self, model, tokenizer, config, input_texts):
        """
        단일 모델로 텍스트 생성
        
        Args:
            model: 모델
            tokenizer: 토크나이저  
            config: 설정
            input_texts: 입력 텍스트 리스트
            
        Returns:
            list: 생성된 텍스트 리스트
        """
        results = []
        
        for text in tqdm(input_texts, desc="텍스트 생성 중"):
            try:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping']
                    )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # 불필요한 토큰 제거
                for token in config['inference']['remove_tokens']:
                    generated_text = generated_text.replace(token, " ")
                
                results.append(generated_text.strip())
                
            except Exception as e:
                log.warning(f"텍스트 생성 중 오류 (fallback 사용): {e}")
                results.append("")  # 빈 문자열로 fallback
        
        return results
    
    def token_level_hard_voting(self, generated_texts_list, reference_tokenizer):
        """
        토큰 단위 하드 보팅
        
        Args:
            generated_texts_list: 각 모델별 생성 텍스트 리스트들
            reference_tokenizer: 기준 토크나이저
            
        Returns:
            list: 앙상블 결과 텍스트 리스트
        """
        ensemble_results = []
        num_samples = len(generated_texts_list[0])
        
        log.info("토큰 단위 하드 보팅 시작...")
        
        for i in tqdm(range(num_samples), desc="앙상블 처리 중"):
            # 각 샘플에 대한 모든 모델의 예측 수집
            texts_for_sample = [texts[i] for texts in generated_texts_list]
            
            # 빈 문자열 제거
            texts_for_sample = [text for text in texts_for_sample if text.strip()]
            
            if not texts_for_sample:
                ensemble_results.append("")
                continue
            
            # 토큰화
            tokenized_texts = []
            for text in texts_for_sample:
                try:
                    tokens = reference_tokenizer.tokenize(text)
                    tokenized_texts.append(tokens)
                except:
                    # 토큰화 실패 시 빈 리스트
                    tokenized_texts.append([])
            
            # 빈 토큰 리스트 제거
            tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
            
            if not tokenized_texts:
                ensemble_results.append("")
                continue
            
            # 최대 길이에 맞춰 정렬
            max_len = max(len(tokens) for tokens in tokenized_texts)
            
            # 각 위치별로 다수결
            final_tokens = []
            for pos in range(max_len):
                tokens_at_pos = []
                for tokens in tokenized_texts:
                    if pos < len(tokens):
                        tokens_at_pos.append(tokens[pos])
                
                if tokens_at_pos:
                    # 가장 많이 선택된 토큰
                    token_counts = Counter(tokens_at_pos)
                    most_common_token = token_counts.most_common(1)[0][0]
                    final_tokens.append(most_common_token)
            
            # 토큰을 텍스트로 변환
            try:
                final_text = reference_tokenizer.convert_tokens_to_string(final_tokens)
                ensemble_results.append(final_text.strip())
            except:
                # 변환 실패 시 가장 첫 번째 텍스트 사용
                ensemble_results.append(texts_for_sample[0])
        
        log.info("토큰 단위 하드 보팅 완료")
        return ensemble_results
    
    def run_ensemble(self, test_data_path):
        """
        하드 보팅 앙상블 실행
        
        Args:
            test_data_path: 테스트 데이터 경로
            
        Returns:
            tuple: (ensemble_result_df, individual_results_list)
        """
        log.info(f"앙상블 추론 시작: {test_data_path}")
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(test_data_path)
        input_texts = test_df['dialogue'].tolist()
        log.info(f"테스트 데이터 로드 완료: {len(input_texts)}개 샘플")
        
        # 각 모델별로 독립적으로 생성
        all_generated_texts = []
        
        for i, (model, tokenizer, config) in enumerate(zip(self.models, self.tokenizers, self.configs)):
            log.info(f"모델 {i+1}/{len(self.models)} 추론 시작...")
            log.info(f"모델 설정 - max_length: {config['inference']['generate_max_length']}, "
                    f"num_beams: {config['inference']['num_beams']}")
            
            generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
            all_generated_texts.append(generated_texts)
            
            log.info(f"모델 {i+1} 추론 완료")
        
        # 하드 보팅으로 앙상블
        log.info("하드 보팅 앙상블 시작...")
        ensemble_results = self.token_level_hard_voting(all_generated_texts, self.tokenizers[0])
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': ensemble_results
        })
        
        log.info("앙상블 추론 완료")
        return result_df, all_generated_texts

def main():
    """앙상블 추론 메인 함수"""
    
    # 사용할 모델 경로들 (수동 지정)
    # TODO: 실제 저장된 모델 경로로 수정 필요
    model_paths = [
        "./models/model_baseline_20250804_123456.zip",  
        "./models/model_baseline_20250804_124512.zip",
        "./models/model_baseline_20250804_125834.zip",
        "./models/model_baseline_20250804_131245.zip"
    ]
    
    # 존재하는 모델 파일만 필터링
    existing_model_paths = []
    for path in model_paths:
        if os.path.exists(path):
            existing_model_paths.append(path)
            log.info(f"모델 파일 확인: {path}")
        else:
            log.warning(f"모델 파일 없음 (건너뜀): {path}")
    
    if not existing_model_paths:
        log.error("사용 가능한 모델 파일이 없습니다!")
        log.info("먼저 WandB sweep을 실행하여 모델을 학습시키세요:")
        log.info("python wandb_sweep.py --count 3")
        return
    
    log.info(f"총 {len(existing_model_paths)}개 모델로 앙상블 진행")
    
    # 앙상블 객체 생성
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ensemble = HardVotingEnsemble(existing_model_paths, device=device)
    
    # 모델들 로딩
    try:
        ensemble.load_models()
    except Exception as e:
        log.error(f"모델 로딩 실패: {e}")
        return
    
    # 앙상블 추론 실행
    test_data_path = "../../input/data/test.csv"
    if not os.path.exists(test_data_path):
        log.error(f"테스트 데이터 파일이 없습니다: {test_data_path}")
        return
    
    try:
        result_df, individual_results = ensemble.run_ensemble(test_data_path)
    except Exception as e:
        log.error(f"앙상블 추론 실패: {e}")
        return
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ensemble_results 폴더 생성
    results_dir = "./ensemble_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 앙상블 결과 저장
    ensemble_path = os.path.join(results_dir, f"ensemble_output_{timestamp}.csv")
    result_df.to_csv(ensemble_path, index=False, encoding='utf-8')
    log.info(f"앙상블 결과 저장: {ensemble_path}")
    
    # 개별 모델 결과들 저장
    test_df = pd.read_csv(test_data_path)
    for i, individual_result in enumerate(individual_results):
        individual_df = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': individual_result
        })
        individual_path = os.path.join(results_dir, f"individual_model_{i+1}_{timestamp}.csv")
        individual_df.to_csv(individual_path, index=False, encoding='utf-8')
        log.info(f"개별 모델 {i+1} 결과 저장: {individual_path}")
    
    # 앙상블 메타데이터 저장
    ensemble_metadata = {
        "timestamp": timestamp,
        "num_models": len(existing_model_paths),
        "model_paths": existing_model_paths,
        "device": device,
        "ensemble_strategy": "token_level_hard_voting",
        "model_metadata": ensemble.metadata_list
    }
    
    metadata_path = os.path.join(results_dir, f"ensemble_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(ensemble_metadata, f, indent=2, ensure_ascii=False)
    log.info(f"앙상블 메타데이터 저장: {metadata_path}")
    
    log.info("=" * 50)
    log.info("앙상블 추론 완료!")
    log.info(f"사용된 모델 수: {len(existing_model_paths)}")
    log.info(f"앙상블 결과: {ensemble_path}")
    log.info("=" * 50)

if __name__ == "__main__":
    main()