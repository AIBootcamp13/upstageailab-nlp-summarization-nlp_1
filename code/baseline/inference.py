#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference_improved.py: baseline.py의 함수들을 재사용하여 정확한 검증점수 재현
"""

import os
import sys
import yaml
import json
import shutil
import zipfile
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '../utils'))

# 로그 유틸리티 임포트
import log_util as log

# baseline.py에서 필요한 함수들 임포트
from baseline import (
    Preprocess,
    DatasetForVal,
    compute_metrics
)

def load_model_and_config_from_zip(zip_path):
    """
    ZIP 파일에서 모델, 토크나이저, 설정을 로드합니다.
    
    Args:
        zip_path (str): 모델 zip 파일 경로
        
    Returns:
        tuple: (model, tokenizer, config)
    """
    log.info(f"ZIP 파일에서 모델 로드 시작: {zip_path}")
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    
    try:
        # ZIP 파일 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        log.info(f"ZIP 파일 압축 해제 완료: {temp_dir}")
        
        # 설정 파일 로드
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info("설정 파일 로드 완료")
        
        # 토크나이저 로드
        tokenizer_path = os.path.join(temp_dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        log.info("토크나이저 로드 완료")
        
        # 모델 로드
        model = BartForConditionalGeneration.from_pretrained(temp_dir)
        model.resize_token_embeddings(len(tokenizer))
        log.info("모델 로드 완료")
        
        return model, tokenizer, config
        
    except Exception as e:
        log.error(f"ZIP 파일에서 모델 로드 중 오류 발생: {e}")
        raise
    finally:
        # 임시 디렉토리 삭제
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            log.info("임시 디렉토리 정리 완료")

def prepare_validation_dataset(config, preprocessor, tokenizer):
    """
    baseline.py와 동일한 방식으로 검증 데이터셋을 준비합니다.
    
    Args:
        config: 설정 딕셔너리
        preprocessor: 데이터 전처리기
        tokenizer: 토크나이저
        
    Returns:
        DatasetForVal: 검증 데이터셋
    """
    log.info("검증 데이터셋 준비 시작")
    
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
    
    # baseline.py와 동일한 DatasetForVal 클래스 사용
    val_inputs_dataset = DatasetForVal(
        val_tokenized_encoder_inputs, 
        val_tokenized_decoder_inputs, 
        val_tokenized_decoder_outputs,
        len(encoder_input_val)
    )
    
    log.info("검증 데이터셋 준비 완료")
    return val_inputs_dataset

def run_validation_inference_with_baseline(config, model, tokenizer):
    """
    baseline.py의 compute_metrics와 Seq2SeqTrainer를 사용한 검증 추론
    
    Args:
        config (dict): 설정 딕셔너리
        model (BartForConditionalGeneration): 로드된 모델
        tokenizer (AutoTokenizer): 로드된 토크나이저
        
    Returns:
        dict: ROUGE 메트릭 결과
    """
    log.info("baseline.py 방식으로 검증 데이터 추론 시작")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 데이터 전처리기 생성 (baseline.py와 동일)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    
    # 검증 데이터셋 준비 (baseline.py와 완전히 동일한 방식)
    val_inputs_dataset = prepare_validation_dataset(config, preprocessor, tokenizer)
    
    # Seq2SeqTrainingArguments 설정 (학습 시와 동일한 파라미터)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./temp_inference_output",
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        seed=config['training']['seed'],
    )
    
    # compute_metrics 함수를 위한 wrapper (baseline.py와 동일)
    def compute_metrics_wrapper(pred):
        return compute_metrics(config, tokenizer, pred)
    
    # Seq2SeqTrainer 생성 (baseline.py와 동일한 방식)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=val_inputs_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper
    )
    
    # 평가 수행 (baseline.py와 완전히 동일한 방식)
    log.info("Seq2SeqTrainer를 사용한 평가 시작")
    eval_results = trainer.evaluate()
    log.info("평가 완료")
    
    # 결과 출력
    log.info("=== baseline.py 방식 검증 결과 ===")
    rouge_results = {}
    for key, value in eval_results.items():
        if 'rouge' in key and key != 'eval_rouge_avg':
            metric_name = key.replace('eval_', '')
            rouge_results[metric_name] = value
            log.info(f"{key}: {value:.6f}")
    
    # 임시 출력 디렉토리 정리
    if os.path.exists("./temp_inference_output"):
        shutil.rmtree("./temp_inference_output", ignore_errors=True)
    
    return rouge_results

def run_test_inference(config, model, tokenizer):
    """
    테스트 데이터에 대한 추론을 수행합니다. (기존 방식 유지)
    
    Args:
        config (dict): 설정 딕셔너리
        model (BartForConditionalGeneration): 로드된 모델
        tokenizer (AutoTokenizer): 로드된 토크나이저
        
    Returns:
        pandas.DataFrame: 추론 결과 (fname, summary)
    """
    log.info("테스트 데이터 추론 시작")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 데이터 전처리기 생성
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    
    # 테스트 데이터 로드
    data_path = config['general']['data_path']
    test_file_path = os.path.join(data_path, 'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    
    # 테스트 데이터셋 준비
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )
    
    from baseline import DatasetForInference
    test_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs,
        test_data['fname'].tolist(),
        len(encoder_input_test)
    )
    
    # DataLoader 생성
    dataloader = DataLoader(
        test_inputs_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )
    
    # 추론 수행
    summary = []
    text_ids = []
    
    model.eval()
    with torch.no_grad():
        for item in dataloader:
            text_ids.extend(item['ID'])
            generated_ids = model.generate(
                input_ids=item['input_ids'].to(device),
                max_length=config['training']['generation_max_length'],
                num_beams=config['inference']['num_beams'],
                early_stopping=config['inference']['early_stopping'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size']
            )
            
            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                summary.append(result)
    
    # 결과 DataFrame 생성
    output = pd.DataFrame({
        "fname": text_ids,
        "summary": summary
    })
    
    # 결과 저장
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    output_path = os.path.join(result_path, "test_output.csv")
    output.to_csv(output_path, index=False)
    log.info(f"테스트 추론 결과를 {output_path}에 저장했습니다.")
    
    return output

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='baseline.py 함수들을 사용한 정확한 검증 추론')
    parser.add_argument('zip_path', type=str, help='모델 zip 파일 경로')
    parser.add_argument('--no-validation', action='store_true', help='검증 데이터 추론 생략')
    parser.add_argument('--no-test', action='store_true', help='테스트 데이터 추론 생략')
    
    args = parser.parse_args()
    
    # ZIP 파일에서 모델, 토크나이저, 설정 로드
    model, tokenizer, config = load_model_and_config_from_zip(args.zip_path)
    
    # 검증 데이터 추론 (baseline.py 방식)
    if not args.no_validation:
        val_metrics = run_validation_inference_with_baseline(config, model, tokenizer)
        log.info("검증 데이터 ROUGE 메트릭 (baseline.py 방식):")
        log.info(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    
    # 테스트 데이터 추론
    if not args.no_test:
        test_output = run_test_inference(config, model, tokenizer)
        log.info("테스트 데이터 추론 결과:")
        log.info(test_output.head())

if __name__ == "__main__":
    main()