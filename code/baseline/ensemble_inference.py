#!/usr/bin/env python3
"""
🔬 대화 요약 앙상블 추론 시스템 (간소화된 함수 기반 버전)

5가지 앙상블 기법을 단순한 함수로 구현하고, 
각 기법 완료 시 즉시 ROUGE 점수를 출력합니다.

앙상블 방식:
1. hard_voting   - 각 모델이 완전한 텍스트 생성 후 토큰별 다수결
2. soft_voting   - 각 모델의 확률 분포를 평균하여 최적 후보 선택
3. length_based  - 각 모델 결과 중 가장 긴 것을 선택
4. logit_level   - 최적화된 Logit 앙상블 (Nucleus Sampling + Beam Search)
5. realtime_token- 매 토큰마다 모든 모델의 확률 분포를 평균하여 생성

사용법:
- python ensemble_inference_simple.py --mode=all           # 모든 방식 비교
- python ensemble_inference_simple.py --mode=hard_voting   # 하드 보팅만
- python ensemble_inference_simple.py --mode=soft_voting   # 소프트 보팅만
- python ensemble_inference_simple.py --mode=length_based  # 길이 기반만
- python ensemble_inference_simple.py --mode=realtime_token # 실시간 토큰 앙상블만
- python ensemble_inference_simple.py --mode=logit_level    # 최적화된 Logit 앙상블만
"""

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append('../utils')
import log_util as log
from baseline import compute_metrics

# 검증 데이터 개수 제한 (None이면 전체 데이터 사용)
DEV_DATA_LIMIT = 50  # 0이나 None이 아닌 정수를 설정하면 해당 개수만큼만 사용

# 테스트 데이터 개수 제한 (None이면 전체 데이터 사용)
TEST_DATA_LIMIT = 50  # 0이나 None이 아닌 정수를 설정하면 해당 개수만큼만 사용

import argparse
import json
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from rouge import Rouge
# baseline.py에서 필요한 클래스들 임포트
from baseline import Preprocess, DatasetForVal, compute_metrics
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import tempfile

# 시드 설정
def set_seed(seed: int = 42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # CUDNN 재현성 설정 (ensemble_inference_best.py와 동일)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 공통 유틸리티 함수들
# =============================================================================

def get_model_paths() -> List[str]:
    """ensemble_inference.py에서 사용했던 정확히 동일한 3개 모델 사용"""
    # ensemble_inference.py에서 지정했던 정확한 모델들
    model_paths = [
        "./models/model_baseline_20250807_224158.zip",  
        "./models/model_baseline_20250807_230111.zip",
    ]
    
    # 존재하는 모델 파일만 필터링
    existing_model_paths = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            existing_model_paths.append(model_path)
            log.info(f"모델 파일 확인: {model_path}")
        else:
            log.warning(f"모델 파일 없음: {model_path}")
    
    if not existing_model_paths:
        log.error("지정된 모델 파일들이 없습니다.")
        return []
    
    log.info(f"총 {len(existing_model_paths)}개 모델로 앙상블 진행")
    return existing_model_paths

def load_models(model_paths: List[str], device: str) -> Tuple[List, List, List, List]:
    """
    모델들을 로드하고 반환 (원래 ensemble_inference.py의 로직 사용)
    
    Returns:
        Tuple[models, tokenizers, configs, metadata_list]
    """
    import yaml
    import shutil
    
    models = []
    tokenizers = []
    configs = []
    metadata_list = []
    
    log.info(f"앙상블 시스템 초기화: {len(model_paths)}개 모델")
    log.info(f"사용 디바이스: {device}")
    log.info("모델들 로딩 시작...")
    
    for i, model_path in enumerate(model_paths):
        log.info(f"모델 {i+1}/{len(model_paths)} 로딩 중: {model_path}")
        
        temp_dir = f"temp_load_{int(time.time())}"
        
        try:
            # 모델 패키지 로딩
            log.info(f"모델 패키지 로딩 시작: {model_path}")
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
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
            
            # 시드 설정 (config에서 - ensemble_inference_best.py와 동일)
            if 'training' in config and 'seed' in config['training']:
                seed = config['training']['seed']
                set_seed(seed)
                log.info(f"모델 로딩 시 시드 설정: {seed}")
            
            # 토크나이저 로드 먼저 (실제 vocab 크기 확인용)
            tokenizer_dir = os.path.join(temp_dir, "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            # Special tokens 추가 (baseline.py와 동일한 방식)
            if 'tokenizer' in config and 'special_tokens' in config['tokenizer']:
                special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
                tokenizer.add_special_tokens(special_tokens_dict)
                log.info(f"Special tokens 추가: {config['tokenizer']['special_tokens']}")
            
            log.info("토크나이저 로드 완료")
            
            # 모델 로드 (temp_dir 루트에 모델 파일들이 있음)
            from transformers import BartConfig
            
            # 저장된 모델의 config.json 직접 로드 및 수정
            model_name = metadata.get('model_info', {}).get('model_name', 'digit82/kobart-summarization')
            bart_config = BartConfig.from_pretrained(model_name)
            actual_vocab_size = len(tokenizer)
            bart_config.vocab_size = actual_vocab_size
            
            model = BartForConditionalGeneration.from_pretrained(temp_dir, config=bart_config)
            
            # 토크나이저와 모델 vocab 크기 맞추기
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model.eval()
            
            log.info(f"BART 설정 로드 완료, vocab_size: {len(tokenizer)}")
            log.info("모델 로드 완료")
            
            models.append(model)
            tokenizers.append(tokenizer)
            configs.append(config)
            metadata_list.append(metadata)
            
            # 임시 파일 정리
            shutil.rmtree(temp_dir)
            
            log.info(f"모델 {i+1} 로딩 완료: {metadata.get('wandb_run_name', 'unknown')} (device: {device})")
            
        except Exception as e:
            log.error(f"모델 {i+1} 로딩 실패: {e}")
            # 임시 파일 정리
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            continue
    
    log.info(f"총 {len(models)}개 모델 로딩 완료")
    return models, tokenizers, configs, metadata_list

def calculate_rouge_scores(predictions: List[str], references: List[str], method_name: str, remove_tokens: List[str]) -> Dict[str, float]:
    """
    ROUGE 점수를 계산하고 즉시 출력
    
    Args:
        predictions: 예측 결과 리스트
        references: 참조 답안 리스트  
        method_name: 앙상블 방법 이름
        remove_tokens: 제거할 토큰들
        
    Returns:
        ROUGE 점수 딕셔너리
    """
    rouge = Rouge()
    
    # 토큰 제거 및 정규화
    cleaned_predictions = []
    cleaned_references = []
    
    for pred, ref in zip(predictions, references):
        # 불필요한 토큰 제거
        pred_clean = pred
        ref_clean = ref
        for token in remove_tokens:
            pred_clean = pred_clean.replace(token, " ")
            ref_clean = ref_clean.replace(token, " ")
        
        # 공백 정리
        pred_clean = " ".join(pred_clean.split()).strip()
        ref_clean = " ".join(ref_clean.split()).strip()
        
        # 빈 문자열 처리
        if not pred_clean.strip():
            pred_clean = "empty"
        if not ref_clean.strip():
            ref_clean = "empty"
            
        cleaned_predictions.append(pred_clean)
        cleaned_references.append(ref_clean)
    
    # ROUGE 계산
    try:
        rouge_results = rouge.get_scores(cleaned_predictions, cleaned_references, avg=True)
        rouge_scores = {key: value["f"] for key, value in rouge_results.items()}
        rouge_scores['rouge-avg'] = (rouge_scores['rouge-1'] + rouge_scores['rouge-2'] + rouge_scores['rouge-l']) / 3
        
        # 즉시 결과 출력
        log.info(f"")
        log.info(f"🎯 {method_name} 완료!")
        log.info(f"📊 ROUGE 점수:")
        log.info(f"   - ROUGE-1: {rouge_scores['rouge-1']:.6f}")
        log.info(f"   - ROUGE-2: {rouge_scores['rouge-2']:.6f}")  
        log.info(f"   - ROUGE-L: {rouge_scores['rouge-l']:.6f}")
        log.info(f"   - ROUGE-avg: {rouge_scores['rouge-avg']:.6f}")
        log.info(f"")
        
        return rouge_scores
        
    except Exception as e:
        log.error(f"ROUGE 계산 실패 ({method_name}): {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge-avg': 0.0}

def save_results_to_json(results: Dict[str, Any], timestamp: str) -> str:
    """결과를 JSON 파일로 저장"""
    results_dir = "./ensemble_results"
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, f"comprehensive_experiment_{timestamp}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    log.info(f"📁 실험 결과 저장: {json_path}")
    return json_path

def calculate_rouge_scores_with_baseline(predictions_ids: List[List[int]], labels_ids: List[List[int]], 
                                         tokenizer, config, method_name: str) -> Dict[str, float]:
    """
    baseline.py의 compute_metrics를 사용하여 ROUGE 점수를 계산하고 즉시 출력
    
    Args:
        predictions_ids: 예측 결과 토큰 ID 리스트
        labels_ids: 참조 답안 토큰 ID 리스트
        tokenizer: 토크나이저
        config: 설정 딕셔너리
        method_name: 앙상블 방법 이름
        
    Returns:
        ROUGE 점수 딕셔너리
    """
    try:
        # 패딩을 사용해 동일한 길이로 맞추기 (수동 패딩 구현)
        
        # 최대 길이 계산
        max_pred_len = max(len(seq) for seq in predictions_ids) if predictions_ids else 1
        max_label_len = max(len(seq) for seq in labels_ids) if labels_ids else 1
        max_len = max(max_pred_len, max_label_len)
        
        # 패딩 적용
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        padded_predictions = []
        padded_labels = []
        
        for seq in predictions_ids:
            if len(seq) < max_len:
                padded_seq = seq + [pad_token_id] * (max_len - len(seq))
            else:
                padded_seq = seq[:max_len]
            padded_predictions.append(padded_seq)
        
        for seq in labels_ids:
            if len(seq) < max_len:
                # labels에는 -100으로 패딩 (compute_metrics가 무시함)
                padded_seq = seq + [-100] * (max_len - len(seq))
            else:
                padded_seq = seq[:max_len]
            padded_labels.append(padded_seq)
        
        # baseline.py 형식으로 데이터 준비
        predictions_array = np.array(padded_predictions)
        labels_array = np.array(padded_labels)
        
        # 예측 객체 생성 (baseline.py의 compute_metrics와 호환)
        pred_object = type('PredictionOutput', (), {
            'predictions': predictions_array,
            'label_ids': labels_array
        })()
        
        # 디버깅: compute_metrics 호출 전 데이터 확인
        log.info(f"🔍 compute_metrics 호출 - predictions shape: {predictions_array.shape}, labels shape: {labels_array.shape}")
        log.info(f"🔍 예측 샘플: {predictions_array[0][:10]} ...")
        log.info(f"🔍 라벨 샘플: {labels_array[0][:10]} ...")
        
        # baseline.py의 compute_metrics 사용
        metrics = compute_metrics(config, tokenizer, pred_object)
        
        # 디버깅: compute_metrics 결과 확인
        log.info(f"🔍 compute_metrics 원본 결과: {metrics}")
        
        # 결과를 앙상블 형식으로 변환
        rouge_scores = {
            'rouge-1': metrics.get('rouge-1', 0.0),
            'rouge-2': metrics.get('rouge-2', 0.0),
            'rouge-l': metrics.get('rouge-l', 0.0)
        }
        rouge_scores['rouge-avg'] = (rouge_scores['rouge-1'] + rouge_scores['rouge-2'] + rouge_scores['rouge-l']) / 3
        
        # 즉시 결과 출력
        log.info(f"")
        log.info(f"🎯 {method_name} 완료!")
        log.info(f"📊 ROUGE 점수:")
        log.info(f"   - ROUGE-1: {rouge_scores['rouge-1']:.6f}")
        log.info(f"   - ROUGE-2: {rouge_scores['rouge-2']:.6f}")
        log.info(f"   - ROUGE-L: {rouge_scores['rouge-l']:.6f}")
        log.info(f"   - ROUGE-avg: {rouge_scores['rouge-avg']:.6f}")
        log.info(f"")
        
        return rouge_scores
        
    except Exception as e:
        log.error(f"ROUGE 계산 실패 ({method_name}): {e}")
        import traceback
        log.error(f"스택 트레이스: {traceback.format_exc()}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge-avg': 0.0}

def convert_text_predictions_to_baseline_format(predictions: List[str], reference_summaries: List[str], 
                                                tokenizer, config, method_name: str) -> Dict[str, float]:
    """
    텍스트 예측 결과를 baseline.py 방식으로 ROUGE 계산하는 공통 함수
    
    Args:
        predictions: 예측 텍스트 리스트
        reference_summaries: 참조 텍스트 리스트
        tokenizer: 토크나이저
        config: 설정 딕셔너리
        method_name: 앙상블 방법 이름
        
    Returns:
        ROUGE 점수 딕셔너리
    """
    predictions_ids = []
    labels_ids = []
    
    # 참조 답안을 토큰 ID로 변환
    for ref_text in reference_summaries:
        ref_tokens = tokenizer(ref_text, return_tensors="pt", truncation=True, padding=False, max_length=512)
        labels_ids.append(ref_tokens['input_ids'][0].tolist())
    
    # 예측 결과를 토큰 ID로 변환
    for pred_text in predictions:
        pred_tokens = tokenizer(pred_text, return_tensors="pt", truncation=True, padding=False, max_length=512)
        predictions_ids.append(pred_tokens['input_ids'][0].tolist())
    
    # baseline.py의 compute_metrics 사용하여 ROUGE 계산
    return calculate_rouge_scores_with_baseline(predictions_ids, labels_ids, tokenizer, config, method_name)

def run_test_inference_for_method(models: List, tokenizers: List, configs: List, method_name: str, test_data: pd.DataFrame, timestamp: str) -> str:
    """
    특정 앙상블 방법으로 테스트 데이터 추론 수행 및 CSV 저장
    
    Args:
        models: 로딩된 모델 리스트
        tokenizers: 토크나이저 리스트  
        configs: 설정 딕셔너리 리스트
        method_name: 앙상블 방법 이름
        test_data: 테스트 데이터프레임
        timestamp: 타임스탬프
        
    Returns:
        str: 저장된 CSV 파일 경로
    """
    log.info(f"🎯 {method_name} 테스트 데이터 추론 시작...")
    
    input_texts = test_data['dialogue'].tolist()
    test_ids = test_data['fname'].tolist()
    predictions = []
    remove_tokens = configs[0]['inference']['remove_tokens']
    
    # 방법별 추론 수행
    if method_name == "hard_voting":
        predictions = test_inference_hard_voting(models, tokenizers, configs, input_texts, remove_tokens)
    elif method_name == "soft_voting":
        predictions = test_inference_soft_voting(models, tokenizers, configs, input_texts, remove_tokens)
    elif method_name == "length_based":
        predictions = test_inference_length_based(models, tokenizers, configs, input_texts, remove_tokens)
    elif method_name == "logit_level":
        predictions = test_inference_logit_level(models, tokenizers, configs, input_texts, remove_tokens)
    elif method_name == "realtime_token_ensemble":
        predictions = test_inference_realtime_token(models, tokenizers, configs, input_texts, remove_tokens)
    else:
        log.error(f"알 수 없는 방법: {method_name}")
        return None
    
    # CSV 저장
    results_dir = "./prediction"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, f"{method_name}_{timestamp}.csv")
    result_df = pd.DataFrame({
        'id': test_ids,
        'summary': predictions
    })
    
    result_df.to_csv(csv_path, index=False, encoding='utf-8')
    log.info(f"📁 {method_name} 테스트 결과 저장: {csv_path}")
    
    return csv_path

# =============================================================================
# 각 앙상블 기법별 평가 함수들
# =============================================================================

def prepare_validation_dataset_for_ensemble(config, preprocessor, tokenizer):
    """
    baseline.py와 동일한 방식으로 검증 데이터셋을 준비합니다.
    
    Args:
        config: 설정 딕셔너리
        preprocessor: 데이터 전처리기
        tokenizer: 토크나이저
        
    Returns:
        DatasetForVal: 검증 데이터셋
    """
    log.info("검증 데이터셋 준비 시작 (baseline.py 방식)")
    
    # 검증 데이터 로드 (baseline.py와 동일한 방식)
    data_path = config['general']['data_path']
    val_file_path = os.path.join(data_path, 'dev.csv')
    val_data = preprocessor.make_set_as_df(val_file_path)
    
    # 입력 데이터 준비 (baseline.py와 동일한 방식)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    
    # 토크나이저 적용 (baseline.py와 완전히 동일한 방식)
    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    
    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    
    val_tokenized_decoder_outputs = tokenizer(
        decoder_output_val, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    
    # 검증 데이터셋 생성 (baseline.py와 동일)
    val_inputs_dataset = DatasetForVal(
        val_tokenized_encoder_inputs, 
        val_tokenized_decoder_inputs, 
        val_tokenized_decoder_outputs,
        len(encoder_input_val)
    )
    
    log.info("검증 데이터셋 준비 완료")
    return val_inputs_dataset

def evaluate_single_model_with_baseline(model, tokenizer, config):
    """
    baseline.py 방식으로 단일 모델 검증 점수 계산
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        config: 설정
        
    Returns:
        dict: ROUGE 메트릭 결과
    """
    log.info("baseline.py 방식으로 검증 점수 계산 시작")
    
    # 데이터 전처리기 생성 (baseline.py와 동일)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    
    # 검증 데이터셋 준비 (baseline.py와 완전히 동일한 방식)
    val_inputs_dataset = prepare_validation_dataset_for_ensemble(config, preprocessor, tokenizer)
    
    # Seq2SeqTrainingArguments 설정 (학습 시와 동일한 파라미터, wandb 비활성화)
    training_args = Seq2SeqTrainingArguments(
        output_dir='./temp_eval_results',
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        predict_with_generate=True,
        generation_max_length=config['inference']['generate_max_length'],
        generation_num_beams=config['inference']['num_beams'],
        include_inputs_for_metrics=False,
        report_to=[],  # wandb 비활성화
        logging_strategy="no"
    )
    
    # Seq2SeqTrainer 생성 (baseline.py와 동일한 방식)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred)
    )
    
    log.info("Seq2SeqTrainer를 사용한 평가 시작")
    # 평가 수행 (baseline.py와 완전히 동일한 방식)
    eval_results = trainer.evaluate(eval_dataset=val_inputs_dataset)
    
    # 임시 디렉토리 정리
    import shutil
    if os.path.exists('./temp_eval_results'):
        shutil.rmtree('./temp_eval_results')
    
    log.info("baseline.py 방식 검증 점수 계산 완료")
    
    # rouge 결과만 추출
    rouge_results = {
        'rouge-1': eval_results['eval_rouge-1'],
        'rouge-2': eval_results['eval_rouge-2'], 
        'rouge-l': eval_results['eval_rouge-l'],
        'rouge-avg': (eval_results['eval_rouge-1'] + eval_results['eval_rouge-2'] + eval_results['eval_rouge-l']) / 3
    }
    
    return rouge_results

def evaluate_individual_models(models: List, tokenizers: List, configs: List, metadata_list: List, val_data: pd.DataFrame) -> List[Dict]:
    """개별 모델들의 성능을 평가 (baseline.py 방식 사용)"""
    log.info("🔍 개별 모델 성능 평가 시작")
    
    individual_scores = []
    
    for i, (model, tokenizer, config, metadata) in enumerate(zip(models, tokenizers, configs, metadata_list)):
        log.info(f"모델 {i+1}/{len(models)} 평가 중...")
        
        # baseline.py 방식으로 정확한 검증 점수 계산
        rouge_scores = evaluate_single_model_with_baseline(model, tokenizer, config)
        
        # ROUGE 점수 즉시 출력
        log.info(f"")
        log.info(f"🎯 개별 모델 {i+1} 완료!")
        log.info(f"📊 ROUGE 점수:")
        log.info(f"   - ROUGE-1: {rouge_scores['rouge-1']:.6f}")
        log.info(f"   - ROUGE-2: {rouge_scores['rouge-2']:.6f}")
        log.info(f"   - ROUGE-L: {rouge_scores['rouge-l']:.6f}")
        log.info(f"   - ROUGE-avg: {rouge_scores['rouge-avg']:.6f}")
        log.info(f"")
        
        individual_scores.append({
            'model_index': i + 1,
            'model_metadata': metadata,
            'rouge_scores': rouge_scores
        })
    
    return individual_scores

def evaluate_hard_voting(models: List, tokenizers: List, configs: List, val_data: pd.DataFrame) -> Dict[str, float]:
    """하드 보팅 앙상블 평가"""
    log.info("🗳️  하드 보팅 앙상블 시작...")
    
    input_texts = val_data['dialogue'].tolist()
    reference_summaries = val_data['summary'].tolist()
    remove_tokens = configs[0]['inference']['remove_tokens']
    tokenizer = tokenizers[0]
    
    # 각 모델별로 토큰 ID 생성
    all_generated_token_ids = []
    for i, (model, model_tokenizer, config) in enumerate(zip(models, tokenizers, configs)):
        model_token_ids = []
        for text in tqdm(input_texts, desc=f"하드보팅 - 모델 {i+1} 토큰 생성"):
            try:
                inputs = model_tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping']
                    )
                
                generated_ids = output_ids[0].tolist()
                model_token_ids.append(generated_ids)
                
            except Exception as e:
                log.warning(f"하드보팅 모델 {i+1} 샘플 생성 실패: {e}")
                model_token_ids.append([])
        
        all_generated_token_ids.append(model_token_ids)
    
    # 하드 보팅으로 앙상블
    predictions = []
    for sample_idx in tqdm(range(len(input_texts)), desc="하드보팅 앙상블 처리"):
        try:
            # 모든 모델의 토큰 ID들을 수집
            all_sequences = []
            for model_idx in range(len(models)):
                if sample_idx < len(all_generated_token_ids[model_idx]):
                    sequence = all_generated_token_ids[model_idx][sample_idx]
                    if sequence:  # 빈 시퀀스가 아니면
                        all_sequences.append(sequence)
            
            if not all_sequences:
                predictions.append("")
                continue
            
            # 가장 긴 시퀀스 길이 찾기
            max_length = max(len(seq) for seq in all_sequences)
            
            # 각 위치별로 다수결 투표
            ensemble_ids = []
            for pos in range(max_length):
                votes = []
                for seq in all_sequences:
                    if pos < len(seq):
                        votes.append(seq[pos])
                
                if votes:
                    # 가장 많이 나온 토큰 선택
                    token_counts = Counter(votes)
                    most_common_token = token_counts.most_common(1)[0][0]
                    ensemble_ids.append(most_common_token)
            
            # 텍스트로 디코딩
            ensemble_text = tokenizer.decode(ensemble_ids, skip_special_tokens=True)
            for token in remove_tokens:
                ensemble_text = ensemble_text.replace(token, " ")
            
            predictions.append(ensemble_text.strip())
            
        except Exception as e:
            log.warning(f"하드보팅 샘플 {sample_idx} 처리 실패: {e}")
            predictions.append("")
    
    # 디버깅: 예측 결과 확인
    log.info(f"🔍 하드 보팅 예측 결과 샘플 (총 {len(predictions)}개):")
    for i in range(min(3, len(predictions))):
        log.info(f"  예측 {i+1}: '{predictions[i]}'")
        log.info(f"  참조 {i+1}: '{reference_summaries[i]}'")
    
    # baseline.py 방식으로 ROUGE 계산
    return convert_text_predictions_to_baseline_format(
        predictions, reference_summaries, tokenizers[0], configs[0], "하드 보팅 앙상블"
    )

def evaluate_soft_voting(models: List, tokenizers: List, configs: List, val_data: pd.DataFrame) -> Dict[str, float]:
    """소프트 보팅 앙상블 평가"""
    log.info("🤝 소프트 보팅 앙상블 시작...")
    
    input_texts = val_data['dialogue'].tolist()
    reference_summaries = val_data['summary'].tolist()
    remove_tokens = configs[0]['inference']['remove_tokens']
    tokenizer = tokenizers[0]
    
    predictions = []
    
    for text in tqdm(input_texts, desc="소프트 보팅 앙상블 처리"):
        try:
            # 각 모델의 beam search 결과들 수집
            all_candidates = []
            all_scores = []
            
            for model, model_tokenizer, config in zip(models, tokenizers, configs):
                inputs = model_tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping'],
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_return_sequences=config['inference']['num_beams']
                    )
                
                # 각 후보별로 텍스트와 점수 저장
                for i, sequence in enumerate(outputs.sequences):
                    text_output = model_tokenizer.decode(sequence, skip_special_tokens=True)
                    for token in remove_tokens:
                        text_output = text_output.replace(token, " ")
                    
                    sequence_score = outputs.sequences_scores[i].item() if hasattr(outputs, 'sequences_scores') else 0.0
                    
                    all_candidates.append(text_output.strip())
                    all_scores.append(sequence_score)
            
            # 점수가 가장 높은 후보 선택
            if all_candidates and all_scores:
                best_idx = np.argmax(all_scores)
                best_candidate = all_candidates[best_idx]
                predictions.append(best_candidate)
            else:
                predictions.append("")
                
        except Exception as e:
            log.warning(f"소프트 보팅 샘플 처리 실패: {e}")
            predictions.append("")
    
    # baseline.py 방식으로 ROUGE 계산
    return convert_text_predictions_to_baseline_format(
        predictions, reference_summaries, tokenizers[0], configs[0], "소프트 보팅 앙상블"
    )

def evaluate_length_based(models: List, tokenizers: List, configs: List, val_data: pd.DataFrame) -> Dict[str, float]:
    """길이 기반 앙상블 평가 (baseline.py 방식 사용)"""
    log.info("📏 길이 기반 앙상블 시작...")
    
    input_texts = val_data['dialogue'].tolist()
    reference_summaries = val_data['summary'].tolist()
    
    predictions_ids = []
    labels_ids = []
    
    # 참조 답안을 토큰 ID로 변환
    tokenizer = tokenizers[0]
    for ref_text in reference_summaries:
        ref_tokens = tokenizer(ref_text, return_tensors="pt", truncation=True, padding=False, max_length=512)
        labels_ids.append(ref_tokens['input_ids'][0].tolist())
    
    for text in tqdm(input_texts, desc="길이 기반 앙상블 처리"):
        try:
            candidates_ids = []
            
            # 각 모델에서 생성
            for model, model_tokenizer, config in zip(models, tokenizers, configs):
                inputs = model_tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping']
                    )
                
                generated_ids = output_ids[0].tolist()
                candidates_ids.append(generated_ids)
            
            # 가장 긴 결과 선택 (토큰 수 기준)
            if candidates_ids:
                longest_candidate = max(candidates_ids, key=len)
                predictions_ids.append(longest_candidate)
            else:
                predictions_ids.append([tokenizer.pad_token_id])
                
        except Exception as e:
            log.warning(f"길이 기반 앙상블 샘플 처리 실패: {e}")
            predictions_ids.append([tokenizer.pad_token_id])
    
    # baseline.py의 compute_metrics 사용하여 ROUGE 계산
    rouge_scores = calculate_rouge_scores_with_baseline(
        predictions_ids, labels_ids, tokenizer, configs[0], "길이 기반 앙상블"
    )
    return rouge_scores

def evaluate_logit_level(models: List, tokenizers: List, configs: List, val_data: pd.DataFrame) -> Dict[str, float]:
    """Logit 레벨 앙상블 평가 (최적화된 Beam Search + Nucleus Sampling)"""
    log.info("🎯 Logit 레벨 앙상블 시작...")
    
    input_texts = val_data['dialogue'].tolist()
    reference_summaries = val_data['summary'].tolist()
    remove_tokens = configs[0]['inference']['remove_tokens']
    tokenizer = tokenizers[0]
    config = configs[0]
    max_length = config['inference']['generate_max_length']
    num_beams = config['inference']['num_beams']
    device = models[0].device
    
    # Nucleus Sampling 파라미터 추가
    top_p = 1.0  # ensemble_inference_best.py와 동일하게 설정 (Nucleus Sampling 비활성화)
    
    predictions = []
    
    for idx, text in enumerate(tqdm(input_texts, desc="Logit 레벨 앙상블 처리")):
        try:
            # 입력 토큰화
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=config['tokenizer']['encoder_max_len'],
                truncation=True,
                padding=True
            ).to(device)
            
            # 각 모델의 encoder 출력 미리 계산
            encoder_outputs_list = []
            for model in models:
                with torch.no_grad():
                    encoder_outputs = model.get_encoder()(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    encoder_outputs_list.append(encoder_outputs.last_hidden_state.clone().detach())
            
            # Beam Search 초기화
            decoder_start_token_id = tokenizer.bos_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = tokenizer.eos_token_id
            batch_size = 1
            beam_size = num_beams
            
            sequences = torch.full((batch_size * beam_size, 1), decoder_start_token_id, device=device)
            beam_scores = torch.zeros(batch_size * beam_size, device=device)
            beam_scores[1:] = -float('inf')
            
            eos_token_id = tokenizer.eos_token_id
            finished_sequences = []
            
            # Beam Search 루프
            for step in range(max_length - 1):
                if len(finished_sequences) >= beam_size:
                    break
                
                current_sequences = sequences[beam_scores > -float('inf')]
                current_scores = beam_scores[beam_scores > -float('inf')]
                
                if len(current_sequences) == 0:
                    break
                
                # 각 모델에서 logits 계산
                all_next_logits = []
                for model_idx, model in enumerate(models):
                    try:
                        with torch.no_grad():
                            decoder_outputs = model.get_decoder()(
                                input_ids=current_sequences,
                                encoder_hidden_states=encoder_outputs_list[model_idx].expand(len(current_sequences), -1, -1),
                                encoder_attention_mask=inputs['attention_mask'].expand(len(current_sequences), -1)
                            )
                            
                            logits = model.lm_head(decoder_outputs.last_hidden_state)
                            next_token_logits = logits[:, -1, :]
                            all_next_logits.append(next_token_logits)
                    except Exception as e:
                        log.warning(f"모델 {model_idx} 스텝 {step} 처리 실패: {e}")
                        continue
                
                # 모든 모델이 실패한 경우 처리
                if not all_next_logits:
                    log.warning(f"샘플 {idx}: 모든 모델이 실패, 빈 문자열 반환")
                    predictions.append("")
                    break
                
                # 모든 모델의 logits 평균
                ensemble_logits = torch.stack(all_next_logits).mean(dim=0)
                
                # Nucleus Sampling (Top-p) 적용
                if top_p < 1.0:
                    for beam_idx in range(ensemble_logits.size(0)):
                        sorted_logits, sorted_indices = torch.sort(ensemble_logits[beam_idx], descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        
                        # 누적 확률이 top_p를 초과하는 토큰들 제거
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        ensemble_logits[beam_idx, indices_to_remove] = -float('inf')
                        
                        # 모든 토큰이 제거된 경우 응급처치 (best.py와 동일)
                        valid_tokens = (ensemble_logits[beam_idx] > -float('inf')).sum().item()
                        if valid_tokens == 0:
                            # 최고 확률 토큰 하나는 유지
                            best_token_idx = torch.argmax(sorted_logits)
                            ensemble_logits[beam_idx, sorted_indices[best_token_idx]] = sorted_logits[best_token_idx]
                
                # Log probabilities 계산
                next_token_log_probs = torch.log_softmax(ensemble_logits, dim=-1)
                
                # 새로운 beam 후보 생성
                vocab_size = next_token_log_probs.size(-1)
                next_scores = current_scores.unsqueeze(1) + next_token_log_probs
                next_scores = next_scores.view(-1)
                
                # Top-k 선택
                top_scores, top_indices = torch.topk(next_scores, min(beam_size * 2, len(next_scores)))
                
                # 새 beam 구성
                new_sequences = []
                new_scores = []
                
                for score, token_idx in zip(top_scores, top_indices):
                    beam_idx = token_idx // vocab_size
                    token_id = token_idx % vocab_size
                    
                    new_seq = torch.cat([
                        current_sequences[beam_idx],
                        torch.tensor([token_id], device=device)
                    ])
                    
                    # EOS 토큰 체크
                    if token_id == eos_token_id:
                        finished_sequences.append((new_seq, score.item()))
                    else:
                        new_sequences.append(new_seq)
                        new_scores.append(score)
                        
                    if len(new_sequences) >= beam_size:
                        break
                
                if not new_sequences:
                    break
                
                # 다음 단계를 위한 업데이트
                max_len = max(len(seq) for seq in new_sequences)
                sequences = torch.full((beam_size, max_len), tokenizer.pad_token_id, device=device)
                beam_scores = torch.full((beam_size,), -float('inf'), device=device)
                
                for i, (seq, score) in enumerate(zip(new_sequences[:beam_size], new_scores[:beam_size])):
                    sequences[i, :len(seq)] = seq
                    beam_scores[i] = score
            
            # 최고 점수 시퀀스 선택
            if finished_sequences:
                best_sequence, best_score = max(finished_sequences, key=lambda x: x[1])
            else:
                best_idx = torch.argmax(beam_scores)
                best_sequence = sequences[best_idx]
            
            # 텍스트로 디코딩
            generated_text = tokenizer.decode(best_sequence, skip_special_tokens=False)
            for token in remove_tokens:
                generated_text = generated_text.replace(token, " ")
            
            predictions.append(generated_text.strip())
            
        except Exception as e:
            log.warning(f"Logit 레벨 앙상블 샘플 {idx} 처리 실패: {e}")
            predictions.append("")
    
    # baseline.py 방식으로 ROUGE 계산
    return convert_text_predictions_to_baseline_format(
        predictions, reference_summaries, tokenizers[0], configs[0], "Logit 레벨 앙상블"
    )

def evaluate_realtime_token(models: List, tokenizers: List, configs: List, val_data: pd.DataFrame) -> Dict[str, float]:
    """실시간 토큰 앙상블 평가"""
    log.info("⚡ 실시간 토큰 앙상블 시작...")
    
    input_texts = val_data['dialogue'].tolist()
    reference_summaries = val_data['summary'].tolist()
    remove_tokens = configs[0]['inference']['remove_tokens']
    tokenizer = tokenizers[0]
    config = configs[0]
    max_length = config['inference']['generate_max_length']
    device = models[0].device
    
    predictions = []
    
    for text in tqdm(input_texts, desc="실시간 토큰 앙상블 처리"):
        try:
            # 입력 토큰화
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=config['tokenizer']['encoder_max_len'],
                truncation=True,
                padding=True
            ).to(device)
            
            # 각 모델의 encoder 출력 미리 계산
            model_encoder_outputs = []
            for model in models:
                with torch.no_grad():
                    encoder_outputs = model.get_encoder()(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    model_encoder_outputs.append(encoder_outputs.last_hidden_state)
            
            # 시작 토큰으로 초기화
            decoder_start_token_id = tokenizer.bos_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = tokenizer.eos_token_id
            generated_sequence = [decoder_start_token_id]
            eos_token_id = tokenizer.eos_token_id
            
            # 토큰별 생성 루프
            for step in range(max_length - 1):
                current_ids = torch.tensor([generated_sequence], device=device)
                
                # 각 모델에서 다음 토큰 logits 계산
                model_logits = []
                for i, model in enumerate(models):
                    try:
                        with torch.no_grad():
                            decoder_outputs = model.get_decoder()(
                                input_ids=current_ids,
                                encoder_hidden_states=model_encoder_outputs[i],
                                encoder_attention_mask=inputs['attention_mask']
                            )
                            
                            logits = model.lm_head(decoder_outputs.last_hidden_state)
                            next_token_logits = logits[0, -1, :]
                            model_logits.append(next_token_logits)
                            
                    except Exception as e:
                        log.warning(f"모델 {i+1} 스텝 {step} 오류: {e}")
                        continue
                
                if not model_logits:
                    break
                
                # 모든 모델의 logits 평균
                ensemble_logits = torch.stack(model_logits).mean(dim=0)
                
                # 다음 토큰 선택 (greedy decoding)
                next_token_id = torch.argmax(ensemble_logits).item()
                
                # EOS 토큰이면 종료
                if next_token_id == eos_token_id:
                    break
                
                generated_sequence.append(next_token_id)
            
            # 텍스트로 디코딩
            generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            for token in remove_tokens:
                generated_text = generated_text.replace(token, " ")
            
            predictions.append(generated_text.strip())
            
        except Exception as e:
            log.warning(f"실시간 토큰 앙상블 샘플 처리 실패: {e}")
            predictions.append("")
    
    # baseline.py 방식으로 ROUGE 계산
    return convert_text_predictions_to_baseline_format(
        predictions, reference_summaries, tokenizers[0], configs[0], "실시간 토큰 앙상블"
    )

# =============================================================================
# 메인 실험 함수
# =============================================================================

def main_comprehensive_experiment():
    """
    🔬 다섯 가지 앙상블 방식 종합 비교 실험
    각 방식이 완료될 때마다 즉시 결과를 출력합니다.
    """
    log.info("🔬 " + "="*60)
    log.info("🎯 다섯 가지 앙상블 방식 종합 비교 실험 시작")
    log.info("="*60)
    
    # 모델 경로 가져오기
    model_paths = get_model_paths()
    if not model_paths:
        log.error("💥 사용 가능한 모델 파일이 없습니다!")
        return
    
    log.info(f"🚀 총 {len(model_paths)}개 모델로 실험 진행")
    
    # 데이터 경로 확인
    val_data_path = "../../input/data/dev.csv"
    if not os.path.exists(val_data_path):
        log.error(f"💥 검증 데이터 없음: {val_data_path}")
        return
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 검증 데이터 로드
    log.info("📊 검증 데이터 로드 중...")
    val_data = pd.read_csv(val_data_path)
    # DEV_DATA_LIMIT이 설정되어 있으면 해당 개수만큼만 사용
    if DEV_DATA_LIMIT is not None and DEV_DATA_LIMIT > 0:
        val_data = val_data.head(DEV_DATA_LIMIT)
    log.info(f"검증 데이터 로드 완료: {len(val_data)}개 샘플")
    
    # 테스트 데이터 로드
    test_data_path = "../../input/data/test.csv"
    test_data = None
    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        # TEST_DATA_LIMIT이 설정되어 있으면 해당 개수만큼만 사용
        if TEST_DATA_LIMIT is not None and TEST_DATA_LIMIT > 0:
            test_data = test_data.head(TEST_DATA_LIMIT)
        log.info(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")
    else:
        log.warning(f"테스트 데이터를 찾을 수 없습니다: {test_data_path}")
    
    # 모델 로딩
    log.info("🤖 모델들 로딩 중...")
    models, tokenizers, configs, metadata_list = load_models(model_paths, device)
    if not models:
        log.error("💥 모델 로딩 실패!")
        return
    
    # 실험 결과 저장용 딕셔너리
    experiment_results = {
        'timestamp': timestamp,
        'model_paths': model_paths,
        'device': device,
        'validation_samples': len(val_data),
        'methods': {},
        'individual_model_scores': [],
        'performance_ranking': [],
        'time_ranking': []
    }
    
    log.info("")
    log.info("🚀 실험 시작! 각 방식이 완료되는 대로 점수를 출력합니다.")
    log.info("="*60)
    
    # 1. 개별 모델 평가
    log.info("1️⃣ 개별 모델 성능 평가")
    log.info("-"*30)
    start_time = time.time()
    individual_scores = evaluate_individual_models(models, tokenizers, configs, metadata_list, val_data)
    individual_time = time.time() - start_time
    experiment_results['individual_model_scores'] = individual_scores
    log.info(f"⏱️  개별 모델 평가 완료: {individual_time:.2f}초")
    log.info("")
    
    # 2. 하드 보팅 앙상블
    log.info("2️⃣ 하드 보팅 앙상블")
    log.info("-"*30)
    start_time = time.time()
    hard_voting_scores = evaluate_hard_voting(models, tokenizers, configs, val_data)
    hard_voting_time = time.time() - start_time
    experiment_results['methods']['hard_voting'] = {
        'rouge_scores': hard_voting_scores,
        'time_seconds': hard_voting_time,
        'method_type': 'Post-processing'
    }
    log.info(f"⏱️  하드 보팅 완료: {hard_voting_time:.2f}초")
    
    # 하드 보팅 테스트 추론
    if test_data is not None:
        csv_path = run_test_inference_for_method(models, tokenizers, configs, "hard_voting", test_data, timestamp)
        experiment_results['methods']['hard_voting']['test_csv_path'] = csv_path
    
    log.info("")
    
    # 3. 소프트 보팅 앙상블
    log.info("3️⃣ 소프트 보팅 앙상블")
    log.info("-"*30)
    start_time = time.time()
    soft_voting_scores = evaluate_soft_voting(models, tokenizers, configs, val_data)
    soft_voting_time = time.time() - start_time
    experiment_results['methods']['soft_voting'] = {
        'rouge_scores': soft_voting_scores,
        'time_seconds': soft_voting_time,
        'method_type': 'Post-processing'
    }
    log.info(f"⏱️  소프트 보팅 완료: {soft_voting_time:.2f}초")
    
    # 소프트 보팅 테스트 추론
    if test_data is not None:
        csv_path = run_test_inference_for_method(models, tokenizers, configs, "soft_voting", test_data, timestamp)
        experiment_results['methods']['soft_voting']['test_csv_path'] = csv_path
    
    log.info("")
    
    # 4. 길이 기반 앙상블
    log.info("4️⃣ 길이 기반 앙상블")
    log.info("-"*30)
    start_time = time.time()
    length_based_scores = evaluate_length_based(models, tokenizers, configs, val_data)
    length_based_time = time.time() - start_time
    experiment_results['methods']['length_based'] = {
        'rouge_scores': length_based_scores,
        'time_seconds': length_based_time,
        'method_type': 'Post-processing'
    }
    log.info(f"⏱️  길이 기반 완료: {length_based_time:.2f}초")
    
    # 길이 기반 테스트 추론
    if test_data is not None:
        csv_path = run_test_inference_for_method(models, tokenizers, configs, "length_based", test_data, timestamp)
        experiment_results['methods']['length_based']['test_csv_path'] = csv_path
    
    log.info("")
    
    # 5. Logit 레벨 앙상블
    log.info("5️⃣ Logit 레벨 앙상블")
    log.info("-"*30)
    start_time = time.time()
    logit_level_scores = evaluate_logit_level(models, tokenizers, configs, val_data)
    logit_level_time = time.time() - start_time
    experiment_results['methods']['logit_level'] = {
        'rouge_scores': logit_level_scores,
        'time_seconds': logit_level_time,
        'method_type': 'Runtime'
    }
    log.info(f"⏱️  Logit 레벨 완료: {logit_level_time:.2f}초")
    
    # Logit 레벨 테스트 추론
    if test_data is not None:
        csv_path = run_test_inference_for_method(models, tokenizers, configs, "logit_level", test_data, timestamp)
        experiment_results['methods']['logit_level']['test_csv_path'] = csv_path
    
    log.info("")
    
    # 6. 실시간 토큰 앙상블
    log.info("6️⃣ 실시간 토큰 앙상블")
    log.info("-"*30)
    start_time = time.time()
    realtime_token_scores = evaluate_realtime_token(models, tokenizers, configs, val_data)
    realtime_token_time = time.time() - start_time
    experiment_results['methods']['realtime_token_ensemble'] = {
        'rouge_scores': realtime_token_scores,
        'time_seconds': realtime_token_time,
        'method_type': 'Runtime'
    }
    log.info(f"⏱️  실시간 토큰 완료: {realtime_token_time:.2f}초")
    
    # 실시간 토큰 테스트 추론
    if test_data is not None:
        csv_path = run_test_inference_for_method(models, tokenizers, configs, "realtime_token_ensemble", test_data, timestamp)
        experiment_results['methods']['realtime_token_ensemble']['test_csv_path'] = csv_path
    
    log.info("")
    
    # 최종 비교 및 순위 정리
    log.info("🏆 최종 실험 결과 요약")
    log.info("="*60)
    
    # 성능 순위 (ROUGE-avg 기준)
    performance_ranking = []
    for method_name, method_data in experiment_results['methods'].items():
        rouge_avg = method_data['rouge_scores']['rouge-avg']
        time_seconds = method_data['time_seconds']
        performance_ranking.append((method_name, rouge_avg, time_seconds))
    
    performance_ranking.sort(key=lambda x: x[1], reverse=True)
    experiment_results['performance_ranking'] = performance_ranking
    
    # 시간 순위 (실행 시간 기준)
    time_ranking = sorted(performance_ranking, key=lambda x: x[2])
    experiment_results['time_ranking'] = time_ranking
    
    # 결과 출력
    log.info("📊 성능 순위 (ROUGE-avg 기준):")
    for i, (method, rouge_avg, time_sec) in enumerate(performance_ranking, 1):
        log.info(f"  {i}위: {method:<20} ROUGE-avg: {rouge_avg:.6f} ({time_sec:.1f}초)")
    
    log.info("")
    log.info("⏱️ 실행 시간 순위:")
    for i, (method, rouge_avg, time_sec) in enumerate(time_ranking, 1):
        log.info(f"  {i}위: {method:<20} {time_sec:.1f}초 (ROUGE-avg: {rouge_avg:.6f})")
    
    # 최고 성능 방법의 CSV를 최종 제출용으로 복사
    if test_data is not None and performance_ranking:
        best_method = performance_ranking[0][0]
        best_csv_path = experiment_results['methods'][best_method].get('test_csv_path')
        if best_csv_path:
            final_csv_path = os.path.join("./prediction", f"best_{best_method}_{timestamp}.csv")
            import shutil
            shutil.copy2(best_csv_path, final_csv_path)
            log.info(f"🏆 최고 성능 방법 ({best_method}) 결과를 최종 제출용으로 저장: {final_csv_path}")
    
    # 결과 저장
    json_path = save_results_to_json(experiment_results, timestamp)
    
    log.info("")
    log.info("✅ 모든 앙상블 실험 완료!")
    log.info(f"📁 결과 파일: {json_path}")
    if test_data is not None:
        log.info("📁 테스트 추론 결과는 ./prediction 폴더에 저장됨")
    log.info("="*60)
    
    return experiment_results

def run_single_method(method_name: str):
    """단일 앙상블 방식 실행"""
    log.info(f"🎯 {method_name} 단일 방식 실행 모드")
    
    # 모델 로딩
    model_paths = get_model_paths()
    if not model_paths:
        return
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models, tokenizers, configs, metadata_list = load_models(model_paths, device)
    if not models:
        return
    
    # 검증 데이터 로드
    val_data_path = "../../input/data/dev.csv"
    if not os.path.exists(val_data_path):
        log.error(f"검증 데이터 없음: {val_data_path}")
        return
    
    val_data = pd.read_csv(val_data_path)
    # DEV_DATA_LIMIT이 설정되어 있으면 해당 개수만큼만 사용
    if DEV_DATA_LIMIT is not None and DEV_DATA_LIMIT > 0:
        val_data = val_data.head(DEV_DATA_LIMIT)
    log.info(f"검증 데이터 로드 완료: {len(val_data)}개 샘플")
    
    # 해당 방식 실행
    start_time = time.time()
    
    if method_name == "hard_voting":
        scores = evaluate_hard_voting(models, tokenizers, configs, val_data)
    elif method_name == "soft_voting":
        scores = evaluate_soft_voting(models, tokenizers, configs, val_data)
    elif method_name == "length_based":
        scores = evaluate_length_based(models, tokenizers, configs, val_data)
    elif method_name == "logit_level":
        scores = evaluate_logit_level(models, tokenizers, configs, val_data)
    elif method_name == "realtime_token":
        scores = evaluate_realtime_token(models, tokenizers, configs, val_data)
    else:
        log.error(f"알 수 없는 방식: {method_name}")
        return
    
    elapsed_time = time.time() - start_time
    log.info(f"✅ {method_name} 실행 완료: {elapsed_time:.2f}초")

# =============================================================================
# 메인 실행부
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='대화 요약 앙상블 추론 (간소화된 함수 기반 버전)')
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['all', 'hard_voting', 'soft_voting', 'length_based', 'realtime_token', 'logit_level'],
        help='실행할 앙상블 방식 선택 (기본값: all - 모든 방식 비교)'
    )
    
    args = parser.parse_args()
    
    log.info("🔬 대화 요약 앙상블 추론 시스템 시작")
    log.info(f"선택된 모드: {args.mode}")
    
    if args.mode == "all":
        main_comprehensive_experiment()
    else:
        run_single_method(args.mode)

# =============================================================================
# 테스트 데이터 추론 함수들
# =============================================================================

def test_inference_hard_voting(models: List, tokenizers: List, configs: List, input_texts: List[str], remove_tokens: List[str]) -> List[str]:
    """하드 보팅 테스트 추론"""
    predictions = []
    
    for i, input_text in enumerate(tqdm(input_texts, desc="하드보팅 테스트 추론")):
        try:
            model_token_ids = []
            
            for model_idx, (model, tokenizer, config) in enumerate(zip(models, tokenizers, configs)):
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping']
                    )
                
                generated_ids = output_ids[0].tolist()
                model_token_ids.append(generated_ids)
            
            # 하드 보팅: 각 위치별 최빈값 선택
            max_len = max(len(ids) for ids in model_token_ids)
            
            ensemble_ids = []
            for pos in range(max_len):
                position_tokens = []
                for ids in model_token_ids:
                    if pos < len(ids):
                        position_tokens.append(ids[pos])
                
                if position_tokens:
                    ensemble_ids.append(max(set(position_tokens), key=position_tokens.count))
            
            # 텍스트 디코딩
            ensemble_text = tokenizers[0].decode(ensemble_ids, skip_special_tokens=True)
            for token in remove_tokens:
                ensemble_text = ensemble_text.replace(token, " ")
            
            predictions.append(ensemble_text.strip())
            
        except Exception as e:
            log.warning(f"하드보팅 테스트 추론 샘플 처리 실패: {e}")
            predictions.append("")
    
    return predictions

def test_inference_soft_voting(models: List, tokenizers: List, configs: List, input_texts: List[str], remove_tokens: List[str]) -> List[str]:
    """소프트 보팅 테스트 추론"""
    predictions = []
    
    for input_text in tqdm(input_texts, desc="소프트보팅 테스트 추론"):
        try:
            all_candidates = []
            all_scores = []
            
            for model, tokenizer, config in zip(models, tokenizers, configs):
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping'],
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_return_sequences=config['inference']['num_beams']
                    )
                
                for i, sequence in enumerate(outputs.sequences):
                    text_output = tokenizer.decode(sequence, skip_special_tokens=True)
                    for token in remove_tokens:
                        text_output = text_output.replace(token, " ")
                    score = outputs.sequences_scores[i].item() if hasattr(outputs, 'sequences_scores') else 0.0
                    all_candidates.append(text_output.strip())
                    all_scores.append(score)
            
            # 점수가 가장 높은 후보 선택
            if all_candidates and all_scores:
                best_idx = int(np.argmax(all_scores))
                predictions.append(all_candidates[best_idx])
            else:
                predictions.append("")
                
        except Exception as e:
            log.warning(f"소프트보팅 테스트 추론 샘플 처리 실패: {e}")
            predictions.append("")
    
    return predictions

def test_inference_length_based(models: List, tokenizers: List, configs: List, input_texts: List[str], remove_tokens: List[str]) -> List[str]:
    """길이 기반 테스트 추론"""
    predictions = []
    
    for input_text in tqdm(input_texts, desc="길이기반 테스트 추론"):
        try:
            candidates_ids = []
            
            for model, tokenizer, config in zip(models, tokenizers, configs):
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=config['inference']['generate_max_length'],
                        num_beams=config['inference']['num_beams'],
                        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                        early_stopping=config['inference']['early_stopping']
                    )
                
                generated_ids = output_ids[0].tolist()
                candidates_ids.append(generated_ids)
            
            # 가장 긴 결과 선택 (토큰 수 기준)
            if candidates_ids:
                longest_ids = max(candidates_ids, key=len)
                generated_text = tokenizers[0].decode(longest_ids, skip_special_tokens=True)
                for token in remove_tokens:
                    generated_text = generated_text.replace(token, " ")
                predictions.append(generated_text.strip())
            else:
                predictions.append("")
                
        except Exception as e:
            log.warning(f"길이기반 테스트 추론 샘플 처리 실패: {e}")
            predictions.append("")
    
    return predictions

def test_inference_logit_level(models: List, tokenizers: List, configs: List, input_texts: List[str], remove_tokens: List[str]) -> List[str]:
    """Logit 레벨 테스트 추론"""
    predictions = []
    
    for input_text in tqdm(input_texts, desc="Logit레벨 테스트 추론"):
        try:
            tokenizer = tokenizers[0]
            config = configs[0]
            device = models[0].device
            
            # 입력 토큰화
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=config['tokenizer']['encoder_max_len'],
                truncation=True,
                padding=True
            ).to(device)
            
            # 각 모델의 encoder 출력 미리 계산
            encoder_outputs_list = []
            for model in models:
                with torch.no_grad():
                    encoder_outputs = model.get_encoder()(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    encoder_outputs_list.append(encoder_outputs.last_hidden_state.clone().detach())
            
            # Beam Search 설정
            max_length = config['inference']['generate_max_length']
            beam_size = config['inference']['num_beams']
            decoder_start_token_id = tokenizer.bos_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = tokenizer.eos_token_id
            eos_token_id = tokenizer.eos_token_id
            
            sequences = torch.full((beam_size, 1), decoder_start_token_id, device=device)
            beam_scores = torch.full((beam_size,), -float('inf'), device=device)
            beam_scores[0] = 0.0
            finished_sequences = []
            
            # Beam Search 루프
            for step in range(max_length - 1):
                valid_mask = beam_scores > -float('inf')
                current_sequences = sequences[valid_mask]
                current_scores = beam_scores[valid_mask]
                if current_sequences.size(0) == 0:
                    break
                
                # 모델별 logits 평균
                all_next_logits = []
                for model_idx, model in enumerate(models):
                    with torch.no_grad():
                        decoder_outputs = model.get_decoder()(
                            input_ids=current_sequences,
                            encoder_hidden_states=encoder_outputs_list[model_idx].expand(current_sequences.size(0), -1, -1),
                            encoder_attention_mask=inputs['attention_mask'].expand(current_sequences.size(0), -1)
                        )
                        logits = model.lm_head(decoder_outputs.last_hidden_state)
                        next_token_logits = logits[:, -1, :]
                        all_next_logits.append(next_token_logits)
                if not all_next_logits:
                    predictions.append("")
                    continue
                ensemble_logits = torch.stack(all_next_logits).mean(dim=0)
                next_token_log_probs = torch.log_softmax(ensemble_logits, dim=-1)
                vocab_size = next_token_log_probs.size(-1)
                next_scores = current_scores.unsqueeze(1) + next_token_log_probs
                next_scores = next_scores.view(-1)
                
                top_k = min(beam_size * 2, next_scores.numel())
                top_scores, top_indices = torch.topk(next_scores, top_k)
                
                new_sequences = []
                new_scores = []
                for score, flat_index in zip(top_scores, top_indices):
                    beam_idx = int(flat_index // vocab_size)
                    token_id = int(flat_index % vocab_size)
                    new_seq = torch.cat([current_sequences[beam_idx], torch.tensor([token_id], device=device)])
                    if eos_token_id is not None and token_id == eos_token_id:
                        finished_sequences.append((new_seq, float(score.item())))
                    else:
                        new_sequences.append(new_seq)
                        new_scores.append(float(score.item()))
                    if len(new_sequences) >= beam_size:
                        break
                if not new_sequences and not finished_sequences:
                    break
                if new_sequences:
                    max_len = max(len(seq) for seq in new_sequences)
                    sequences = torch.full((beam_size, max_len), tokenizer.pad_token_id, device=device)
                    beam_scores = torch.full((beam_size,), -float('inf'), device=device)
                    for i, (seq, score) in enumerate(zip(new_sequences[:beam_size], new_scores[:beam_size])):
                        sequences[i, :len(seq)] = seq
                        beam_scores[i] = score
                else:
                    break
            
            # 최고 점수 시퀀스 선택
            if finished_sequences:
                best_sequence, _ = max(finished_sequences, key=lambda x: x[1])
            else:
                best_idx = int(torch.argmax(beam_scores))
                best_sequence = sequences[best_idx]
            
            generated_text = tokenizer.decode(best_sequence, skip_special_tokens=False)
            for token in remove_tokens:
                generated_text = generated_text.replace(token, " ")
            predictions.append(generated_text.strip())
        except Exception as e:
            log.warning(f"Logit레벨 테스트 추론 샘플 처리 실패: {e}")
            predictions.append("")
    
    return predictions

def test_inference_realtime_token(models: List, tokenizers: List, configs: List, input_texts: List[str], remove_tokens: List[str]) -> List[str]:
    """실시간 토큰 테스트 추론"""
    predictions = []
    
    for input_text in tqdm(input_texts, desc="실시간토큰 테스트 추론"):
        try:
            inputs = tokenizers[0](
                input_text,
                return_tensors="pt",
                max_length=configs[0]['tokenizer']['encoder_max_len'],
                truncation=True,
                padding=True
            ).to(models[0].device)
            
            max_length = configs[0]['inference']['generate_max_length']
            sequences = inputs['input_ids'].clone()
            device = inputs['input_ids'].device
            
            # 토큰별 생성
            for step in range(max_length - sequences.size(1)):
                model_predictions = []
                
                # 각 모델의 다음 토큰 예측
                for model in models:
                    with torch.no_grad():
                        outputs = model(input_ids=sequences)
                        next_token_logits = outputs.logits[0, -1, :]
                        next_token = torch.argmax(next_token_logits).item()
                        model_predictions.append(next_token)
                
                # 다수결 투표
                if model_predictions:
                    ensemble_token = max(set(model_predictions), key=model_predictions.count)
                    sequences = torch.cat([sequences, torch.tensor([[ensemble_token]], device=device)], dim=1)
                    
                    # EOS 토큰 체크
                    if ensemble_token == tokenizers[0].eos_token_id:
                        break
                else:
                    break
            
            # 생성된 부분만 디코딩
            generated_sequence = sequences[0][inputs['input_ids'].size(1):]
            generated_text = tokenizers[0].decode(generated_sequence, skip_special_tokens=True)
            
            for token in remove_tokens:
                generated_text = generated_text.replace(token, " ")
            
            predictions.append(generated_text.strip())
            
        except Exception as e:
            log.warning(f"실시간토큰 테스트 추론 샘플 처리 실패: {e}")
            predictions.append("")
    
    return predictions

if __name__ == "__main__":
    main()