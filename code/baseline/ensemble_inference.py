#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Inference System for Multiple Strategies
WandB sweep으로 생성된 여러 모델들을 다양한 앙상블 방식으로 추론하는 시스템

지원하는 앙상블 방식:
1. 하드 보팅 (Hard Voting): 토큰별 다수결
2. 소프트 보팅 (Soft Voting): 확률 분포 평균  
3. 길이 기반 (Length-based): 가장 긴 결과 선택
4. 실시간 토큰 앙상블 (Realtime Token Ensemble): 매 토큰마다 확률 분포 평균

사용법:
- python ensemble_inference.py --mode=all           # 모든 방식 비교
- python ensemble_inference.py --mode=hard_voting   # 하드 보팅만
- python ensemble_inference.py --mode=soft_voting   # 소프트 보팅만
- python ensemble_inference.py --mode=length_based  # 길이 기반만
- python ensemble_inference.py --mode=realtime_token # 실시간 토큰 앙상블만
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
import random
import numpy as np

from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

# baseline.py에서 필요한 클래스들 임포트
from baseline import Preprocess, DatasetForVal, compute_metrics
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import tempfile

def get_model_paths():
    """
    모델 경로들을 반환하는 공통 함수
    
    Returns:
        list: 존재하는 모델 파일 경로들
    """
    # TODO: 실제 저장된 모델 경로로 수정 필요
    model_paths = [
        "./models/model_baseline_20250804_063540.zip",  
        "./models/model_baseline_20250804_064025.zip",
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
        return []
    
    log.info(f"총 {len(existing_model_paths)}개 모델 파일 확인됨")
    return existing_model_paths

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
        
        # 시드 설정 (재현성 보장)
        if 'training' in config and 'seed' in config['training']:
            seed = config['training']['seed']
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
        
        # 저장된 모델의 config.json 직접 로드 및 수정
        try:
            # 저장된 config.json 파일 읽기
            config_path = os.path.join(temp_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    model_config_dict = json.load(f)
                
                # 분류 관련 설정 제거
                model_config_dict.pop('num_labels', None)
                model_config_dict.pop('id2label', None)
                model_config_dict.pop('label2id', None)
                
                # 수정된 config로 BartConfig 생성
                bart_config = BartConfig(**model_config_dict)
                log.info(f"BART 설정 로드 완료 (config.json 사용), vocab_size: {bart_config.vocab_size}")
            else:
                # config.json이 없으면 기본 방식 사용
                model_name = config['general']['model_name']
                bart_config = BartConfig.from_pretrained(model_name)
                actual_vocab_size = len(tokenizer)
                bart_config.vocab_size = actual_vocab_size
                log.info(f"BART 설정 로드 완료 (기본 방식), vocab_size: {actual_vocab_size}")
        except Exception as e:
            log.warning(f"config.json 처리 중 오류, 기본 방식 사용: {e}")
            model_name = config['general']['model_name']
            bart_config = BartConfig.from_pretrained(model_name)
            actual_vocab_size = len(tokenizer)
            bart_config.vocab_size = actual_vocab_size
        
        # 모델 로드 (config의 vocab_size가 조정된 상태)
        model = BartForConditionalGeneration.from_pretrained(temp_dir, config=bart_config)
        
        # 토큰 임베딩 크기 조정 (필수! wandb_sweep.py와 동일하게 처리)
        # special tokens가 추가된 경우 반드시 필요
        model.resize_token_embeddings(len(tokenizer))
        model.eval()  # evaluation 모드 설정
        log.info("모델 로드 완료")
        
        return model, tokenizer, config, metadata
        
    except Exception as e:
        log.error(f"모델 패키지 로딩 중 오류: {e}")
        log.error(f"오류 세부 정보: {type(e).__name__}")
        if "num_labels" in str(e) and "id2label" in str(e):
            log.error("히트: BART 모델 설정 불일치 문제입니다. 모델을 다시 학습하거나 config 설정을 확인하세요.")
        raise
        
    finally:
        # 임시 폴더 삭제
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

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
    
    # baseline.py와 동일한 DatasetForVal 클래스 사용
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
    temp_dir = tempfile.mkdtemp()
    training_args = Seq2SeqTrainingArguments(
        output_dir=temp_dir,
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        seed=config['training']['seed'],
        report_to=[],  # wandb 비활성화
        logging_strategy="no",  # 로깅 비활성화
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
    
    # 결과 추출
    rouge_results = {}
    for key, value in eval_results.items():
        if 'rouge' in key and key != 'eval_rouge_avg':
            metric_name = key.replace('eval_', '')
            rouge_results[metric_name] = value
    
    # rouge-avg 계산 추가
    if 'rouge-1' in rouge_results and 'rouge-2' in rouge_results and 'rouge-l' in rouge_results:
        rouge_avg = (rouge_results['rouge-1'] + rouge_results['rouge-2'] + rouge_results['rouge-l']) / 3
        rouge_results['rouge-avg'] = rouge_avg
    
    # 임시 디렉토리 정리
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return rouge_results

class RealtimeTokenEnsemble:
    """
    실시간 토큰 단위 앙상블 클래스
    각 스텝마다 모든 모델에서 다음 토큰 확률 분포를 획득하여 앙상블
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
        
        log.info(f"실시간 토큰 앙상블 시스템 초기화: {len(model_paths)}개 모델")
        log.info(f"사용 디바이스: {device}")
    
    def load_models(self):
        """모든 모델들을 로딩"""
        log.info("모델들 로딩 시작...")
        
        for i, path in enumerate(self.model_paths):
            log.info(f"모델 {i+1}/{len(self.model_paths)} 로딩 중: {path}")
            
            try:
                model, tokenizer, config, metadata = load_model_package(path)
                
                # GPU 메모리 확인 및 모델 로딩
                try:
                    model.to(self.device)
                    model.eval()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log.error(f"GPU 메모리 부족으로 모델 {i+1} 로딩 실패. CPU로 fallback 시도...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # GPU 메모리 정리
                        self.device = "cpu"
                        model.to(self.device)
                        model.eval()
                        log.warning(f"모델 {i+1}을 CPU에서 실행합니다. 성능이 느려질 수 있습니다.")
                    else:
                        raise
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.configs.append(config)
                self.metadata_list.append(metadata)
                
                log.info(f"모델 {i+1} 로딩 완료: {metadata.get('wandb_run_name', 'Unknown')} (device: {self.device})")
                
            except Exception as e:
                log.error(f"모델 {i+1} 로딩 실패: {e}")
                log.error(f"경로: {path}")
                raise
        
        log.info(f"총 {len(self.models)}개 모델 로딩 완료")
    
    def generate_ensemble_sequence_single(self, input_text, config):
        """
        단일 텍스트에 대한 실시간 토큰 앙상블 (개선된 로직)
        
        Args:
            input_text: 단일 입력 텍스트
            config: 생성 설정
            
        Returns:
            str: 생성된 텍스트
        """
        tokenizer = self.tokenizers[0]
        max_length = config['inference']['generate_max_length']
        
        # 입력 토큰화
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=config['tokenizer']['encoder_max_len'],
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 🚀 핵심 최적화: 각 모델의 encoder output을 한 번만 계산
        model_encoder_outputs = []
        for model in self.models:
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask']
                )
                model_encoder_outputs.append(encoder_outputs.last_hidden_state)
        
        # 디코더 시작 토큰
        decoder_start_token_id = tokenizer.bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.eos_token_id
        
        # 생성된 시퀀스 (시작 토큰으로 초기화)
        generated_sequence = [decoder_start_token_id]
        eos_token_id = tokenizer.eos_token_id
        
        # 🔄 토큰별 생성 루프
        for step in range(max_length - 1):
            # 현재까지의 시퀀스를 텐서로 변환
            current_ids = torch.tensor([generated_sequence], device=self.device)
            
            # 각 모델에서 다음 토큰 logits 계산
            model_logits = []
            successful_models = 0
            
            for i, model in enumerate(self.models):
                try:
                    with torch.no_grad():
                        # 디코더 실행 (미리 계산된 encoder output 사용)
                        decoder_outputs = model.get_decoder()(
                            input_ids=current_ids,
                            encoder_hidden_states=model_encoder_outputs[i],
                            encoder_attention_mask=inputs['attention_mask']
                        )
                        
                        # LM head로 vocabulary logits 계산
                        logits = model.lm_head(decoder_outputs.last_hidden_state)
                        next_token_logits = logits[0, -1, :]  # 마지막 위치의 logits
                        
                        model_logits.append(next_token_logits)
                        successful_models += 1
                        
                except Exception as e:
                    log.warning(f"모델 {i+1} 스텝 {step} 오류: {e}")
                    continue
            
            if successful_models == 0:
                log.error(f"스텝 {step}: 모든 모델 실패")
                break
            
            # 🧮 성공한 모델들의 logits 평균 계산
            if len(model_logits) > 1:
                ensemble_logits = torch.stack(model_logits).mean(dim=0)
            else:
                ensemble_logits = model_logits[0]
            
            # 🎯 Greedy decoding: 가장 높은 확률의 토큰 선택
            next_token_id = torch.argmax(ensemble_logits).item()
            
            # 생성된 토큰을 시퀀스에 추가
            generated_sequence.append(next_token_id)
            
            # ✅ EOS 토큰 도달 시 생성 종료
            if next_token_id == eos_token_id:
                log.debug(f"EOS 도달: 스텝 {step}, 길이 {len(generated_sequence)}")
                break
        
        # 🔤 텍스트로 디코딩
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # 불필요한 토큰 제거
        for token in config['inference']['remove_tokens']:
            generated_text = generated_text.replace(token, " ")
            
        return generated_text.strip()
    
    def generate_ensemble_sequence(self, input_ids, config):
        """
        배치 처리를 위한 래퍼 (하위 호환성)
        
        Args:
            input_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크  
            config: 생성 설정
            
        Returns:
            torch.Tensor: 생성된 시퀀스
        """
        # 단일 텍스트 처리로 위임
        tokenizer = self.tokenizers[0]
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated_text = self.generate_ensemble_sequence_single(input_text, config)
        
        # 다시 토큰화하여 반환
        generated_ids = tokenizer(
            generated_text, 
            return_tensors="pt",
            add_special_tokens=False
        )['input_ids']
        
        return generated_ids
    
    def generate_with_realtime_ensemble(self, input_texts, config):
        """
        실시간 앙상블로 텍스트 생성 (개선된 버전)
        
        Args:
            input_texts: 입력 텍스트 리스트
            config: 생성 설정
            
        Returns:
            list: 생성된 텍스트 리스트
        """
        results = []
        
        log.info(f"Realtime Token Ensemble 시작: {len(input_texts)}개 텍스트")
        
        for i, text in enumerate(tqdm(input_texts, desc="실시간 토큰 앙상블 처리 중")):
            try:
                # 🚀 개선된 단일 텍스트 처리 사용
                generated_text = self.generate_ensemble_sequence_single(text, config)
                results.append(generated_text)
                
                # 진행 상황 로깅 (매 10개마다)
                if (i + 1) % 10 == 0:
                    log.info(f"진행 상황: {i+1}/{len(input_texts)} 완료")
                
            except Exception as e:
                log.warning(f"텍스트 {i+1} 실시간 앙상블 오류: {e}")
                results.append("")  # 빈 문자열로 fallback
        
        log.info("Realtime Token Ensemble 완료")
        return results
    
    def generate_with_single_model(self, model, tokenizer, config, input_texts):
        """
        비교를 위한 단일 모델 텍스트 생성
        
        Args:
            model: 모델
            tokenizer: 토크나이저  
            config: 설정
            input_texts: 입력 텍스트 리스트
            
        Returns:
            list: 생성된 텍스트 리스트
        """
        results = []
        
        for text in tqdm(input_texts, desc="단일 모델 텍스트 생성 중"):
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
    
    def evaluate_on_validation(self, val_data_path):
        """
        실시간 앙상블 검증 데이터 평가
        
        Args:
            val_data_path: 검증 데이터 경로
            
        Returns:
            dict: 평가 결과
        """
        import time
        
        log.info(f"Realtime Token Ensemble 검증 데이터 평가 시작: {val_data_path}")
        
        # 검증 데이터 로드
        try:
            val_df = pd.read_csv(val_data_path)
            
            # 필수 컬럼 존재 확인
            required_columns = ['dialogue', 'summary']
            for col in required_columns:
                if col not in val_df.columns:
                    log.error(f"검증 데이터에 필수 컬럼 '{col}'이 없습니다. 사용 가능한 컬럼: {list(val_df.columns)}")
                    return None
            
            val_df_sample = val_df.head(50)  # 빠른 테스트를 위해 50개만
            input_texts = val_df_sample['dialogue'].tolist()
            reference_summaries = val_df_sample['summary'].tolist()
            
            # 빈 데이터 확인
            if not input_texts or not reference_summaries:
                log.error("검증 데이터가 비어있습니다.")
                return None
                
            log.info(f"검증 데이터 로드 완료: {len(input_texts)}개 샘플")
        except FileNotFoundError:
            log.error(f"검증 데이터 파일을 찾을 수 없습니다: {val_data_path}")
            return None
        except pd.errors.EmptyDataError:
            log.error(f"검증 데이터 파일이 비어있습니다: {val_data_path}")
            return None
        except Exception as e:
            log.error(f"검증 데이터 로드 실패: {e}")
            return None
        
        # 시간 측정 시작
        start_time = time.time()
        
        # Realtime Token Ensemble으로 생성
        log.info("Realtime Token Ensemble 생성 시작...")
        realtime_results = self.generate_with_realtime_ensemble(input_texts, self.configs[0])
        
        generation_time = time.time() - start_time
        log.info(f"Realtime Token Ensemble 생성 완료: {generation_time:.2f}초")
        
        # ROUGE 점수 계산
        def calculate_rouge_scores(predictions, references, method_name):
            from rouge import Rouge
            rouge = Rouge()
            
            # 불필요한 토큰 제거
            cleaned_predictions = []
            cleaned_references = []
            for pred, ref in zip(predictions, references):
                pred_clean = pred.strip()
                ref_clean = ref.strip()
                for token in self.configs[0]['inference']['remove_tokens']:
                    pred_clean = pred_clean.replace(token, " ")
                    ref_clean = ref_clean.replace(token, " ")
                pred_clean = pred_clean.strip() if pred_clean.strip() else "empty"
                ref_clean = ref_clean.strip() if ref_clean.strip() else "empty"
                cleaned_predictions.append(pred_clean)
                cleaned_references.append(ref_clean)
            
            try:
                rouge_results = rouge.get_scores(cleaned_predictions, cleaned_references, avg=True)
                rouge_scores = {key: value["f"] for key, value in rouge_results.items()}
                # rouge-avg 계산 추가
                rouge_avg = (rouge_scores['rouge-1'] + rouge_scores['rouge-2'] + rouge_scores['rouge-l']) / 3
                rouge_scores['rouge-avg'] = rouge_avg
                
                log.info(f"{method_name} 검증 점수 - ROUGE-1: {rouge_scores['rouge-1']:.4f}, "
                        f"ROUGE-2: {rouge_scores['rouge-2']:.4f}, ROUGE-L: {rouge_scores['rouge-l']:.4f}, "
                        f"ROUGE-avg: {rouge_scores['rouge-avg']:.4f}")
                return rouge_scores
            except Exception as e:
                log.warning(f"{method_name} ROUGE 계산 오류: {e}")
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge-avg': 0.0}
        
        realtime_scores = calculate_rouge_scores(realtime_results, reference_summaries, "Realtime Token Ensemble")
        
        evaluation_results = {
            'realtime_token_ensemble_scores': realtime_scores,
            'generation_time_seconds': generation_time,
            'num_validation_samples': len(input_texts)
        }
        
        log.info("Realtime Token Ensemble 검증 데이터 평가 완료")
        return evaluation_results
    
    def run_ensemble(self, test_data_path):
        """
        Realtime Token Ensemble 실행
        
        Args:
            test_data_path: 테스트 데이터 경로
            
        Returns:
            tuple: (ensemble_result_df, generation_time)
        """
        import time
        
        log.info(f"Realtime Token Ensemble 추론 시작: {test_data_path}")
        
        # 테스트 데이터 로드
        try:
            test_df = pd.read_csv(test_data_path)
            test_df_sample = test_df.head(20)  # 빠른 테스트용
            input_texts = test_df_sample['dialogue'].tolist()
            log.info(f"테스트 데이터 로드 완료: {len(input_texts)}개 샘플")
        except Exception as e:
            log.error(f"테스트 데이터 로드 실패: {e}")
            return None, 0
        
        # 시간 측정
        start_time = time.time()
        
        # Realtime Token Ensemble 실행
        realtime_results = self.generate_with_realtime_ensemble(input_texts, self.configs[0])
        
        generation_time = time.time() - start_time
        log.info(f"Realtime Token Ensemble 완료: {generation_time:.2f}초")
        
        # 결과 데이터프레임 생성
        realtime_df = pd.DataFrame({
            'fname': test_df_sample['fname'],
            'summary': realtime_results
        })
        
        return realtime_df, generation_time

def main_comprehensive_experiment():
    """
    🔬 네 가지 앙상블 방식 종합 비교 실험
    
    1. 하드 보팅 (Token-level Hard Voting)
    2. 소프트 보팅 (Probability-based Soft Voting) 
    3. 길이 기반 (Length-based Selection)
    4. 실시간 토큰 앙상블 (Realtime Token Ensemble)
    """
    import time
    
    log.info("🔬 " + "="*60)
    log.info("🎯 네 가지 앙상블 방식 종합 비교 실험 시작")
    log.info("="*60)
    
    # 공통 함수로 모델 경로 가져오기
    existing_model_paths = get_model_paths()
    if not existing_model_paths:
        log.error("💥 사용 가능한 모델 파일이 없습니다!")
        return
    
    log.info(f"🚀 총 {len(existing_model_paths)}개 모델로 실험 진행")
    
    # 데이터 경로 확인
    val_data_path = "../../input/data/dev.csv"
    test_data_path = "../../input/data/test.csv"
    
    if not os.path.exists(val_data_path):
        log.error(f"💥 검증 데이터 없음: {val_data_path}")
        return
    if not os.path.exists(test_data_path):
        log.error(f"💥 테스트 데이터 없음: {test_data_path}")
        return
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 실험 결과를 저장할 딕셔너리
    experiment_results = {
        'timestamp': timestamp,
        'model_paths': existing_model_paths,
        'device': device,
        'methods': {},
        'performance_ranking': [],
        'time_ranking': []
    }
    
    # 📊 실험 1: HardVotingEnsemble의 세 가지 방식
    log.info("\n" + "🔥 " + "="*50)
    log.info("📊 실험 1: Post-processing 앙상블 방식들")
    log.info("="*50)
    
    hard_ensemble = PostProcessingEnsemble(existing_model_paths, device=device)
    hard_ensemble.load_models()
    
    # 검증 데이터 평가
    start_time = time.time()
    hard_evaluation = hard_ensemble.evaluate_on_validation(val_data_path)
    hard_time = time.time() - start_time
    
    if hard_evaluation:
        experiment_results['methods']['hard_voting'] = {
            'rouge_scores': hard_evaluation['hard_voting_scores'],
            'time_seconds': hard_time,
            'method_type': 'Post-processing'
        }
        experiment_results['methods']['soft_voting'] = {
            'rouge_scores': hard_evaluation['soft_voting_scores'],
            'time_seconds': hard_time,  # 같은 실행에서 나온 결과
            'method_type': 'Post-processing'
        }
        experiment_results['methods']['length_based'] = {
            'rouge_scores': hard_evaluation['length_based_scores'],
            'time_seconds': hard_time,  # 같은 실행에서 나온 결과  
            'method_type': 'Post-processing'
        }
    
    # 📊 실험 2: RealtimeTokenEnsemble
    log.info("\n" + "🔥 " + "="*50)
    log.info("⚡ 실험 2: 실시간 토큰 앙상블")
    log.info("="*50)
    
    try:
        realtime_ensemble = RealtimeTokenEnsemble(existing_model_paths, device=device)
        realtime_ensemble.load_models()
        
        # 검증 데이터 평가
        start_time = time.time()
        realtime_evaluation = realtime_ensemble.evaluate_on_validation(val_data_path)
        realtime_time = time.time() - start_time
        
        if realtime_evaluation:
            experiment_results['methods']['realtime_token_ensemble'] = {
                'rouge_scores': realtime_evaluation['realtime_token_ensemble_scores'],
                'time_seconds': realtime_time,
                'method_type': 'Runtime'
            }
    except Exception as e:
        log.error(f"실시간 토큰 앙상블 실행 오류: {e}")
        experiment_results['methods']['realtime_token_ensemble'] = {
            'error': str(e),
            'method_type': 'Runtime'
        }
    
    # 📈 성능 순위 분석
    log.info("\n" + "🏆 " + "="*50)
    log.info("📈 종합 성능 분석 결과")
    log.info("="*50)
    
    # ROUGE-avg 기준 성능 순위
    performance_data = []
    for method_name, method_data in experiment_results['methods'].items():
        if 'rouge_scores' in method_data:
            rouge_avg = method_data['rouge_scores']['rouge-avg']
            time_taken = method_data['time_seconds']
            performance_data.append((method_name, rouge_avg, time_taken))
    
    if performance_data:
        # 성능순 정렬
        performance_data.sort(key=lambda x: x[1], reverse=True)
        experiment_results['performance_ranking'] = performance_data
        
        # 속도순 정렬
        time_data = sorted(performance_data, key=lambda x: x[2])
        experiment_results['time_ranking'] = time_data
        
        log.info("🥇 성능 순위 (ROUGE-avg 기준):")
        for i, (method, rouge_avg, time_taken) in enumerate(performance_data, 1):
            method_type = experiment_results['methods'][method]['method_type']
            log.info(f"  {i}위. {method}: {rouge_avg:.4f} ({time_taken:.1f}초, {method_type})")
        
        log.info("\n⚡ 속도 순위:")
        for i, (method, rouge_avg, time_taken) in enumerate(time_data, 1):
            method_type = experiment_results['methods'][method]['method_type']
            log.info(f"  {i}위. {method}: {time_taken:.1f}초 (ROUGE-avg: {rouge_avg:.4f})")
        
        # 📊 상세 점수 출력
        log.info("\n📊 상세 ROUGE 점수:")
        for method_name, method_data in experiment_results['methods'].items():
            if 'rouge_scores' in method_data:
                scores = method_data['rouge_scores']
                log.info(f"\n🔹 {method_name}:")
                log.info(f"   ROUGE-1: {scores['rouge-1']:.4f}")
                log.info(f"   ROUGE-2: {scores['rouge-2']:.4f}")
                log.info(f"   ROUGE-L: {scores['rouge-l']:.4f}")
                log.info(f"   ROUGE-avg: {scores['rouge-avg']:.4f}")
                log.info(f"   실행시간: {method_data['time_seconds']:.1f}초")
        
        # 🎯 최적 방식 추천
        best_performance = performance_data[0]
        fastest_method = time_data[0]
        
        log.info("\n" + "🎯 " + "="*50)
        log.info("💡 추천 결과")
        log.info("="*50)
        log.info(f"🏆 최고 성능: {best_performance[0]} (ROUGE-avg: {best_performance[1]:.4f})")
        log.info(f"⚡ 최고 속도: {fastest_method[0]} ({fastest_method[2]:.1f}초)")
        
        # 성능 vs 속도 trade-off 분석
        performance_gap = best_performance[1] - fastest_method[1] 
        speed_ratio = fastest_method[2] / best_performance[2]
        
        if performance_gap < 0.01 and speed_ratio < 0.5:
            log.info(f"💎 추천: {fastest_method[0]} (성능 차이 미미하고 속도 우수)")
        elif performance_gap > 0.02:
            log.info(f"💎 추천: {best_performance[0]} (성능 차이 유의미)")
        else:
            log.info("💭 성능과 속도를 고려하여 용도에 맞게 선택하세요")
    
    # 결과 저장
    results_dir = "./ensemble_results"
    os.makedirs(results_dir, exist_ok=True)
    
    experiment_metadata_path = os.path.join(results_dir, f"comprehensive_experiment_{timestamp}.json")
    with open(experiment_metadata_path, "w", encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    log.info(f"\n💾 실험 결과 저장: {experiment_metadata_path}")
    
    log.info("\n" + "🎉 " + "="*50)
    log.info("✅ 종합 비교 실험 완료!")
    log.info("="*50)
    
    return experiment_results

class PostProcessingEnsemble:
    """
    후처리 기반 앙상블 추론 클래스
    각 모델이 독립적으로 완전한 텍스트를 생성한 후 다양한 방식으로 앙상블:
    - 하드 보팅: 토큰 단위 다수결
    - 소프트 보팅: 확률 분포 평균
    - 길이 기반: 가장 긴 결과 선택
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
                
                # GPU 메모리 확인 및 모델 로딩
                try:
                    model.to(self.device)
                    model.eval()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log.error(f"GPU 메모리 부족으로 모델 {i+1} 로딩 실패. CPU로 fallback 시도...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # GPU 메모리 정리
                        self.device = "cpu"
                        model.to(self.device)
                        model.eval()
                        log.warning(f"모델 {i+1}을 CPU에서 실행합니다. 성능이 느려질 수 있습니다.")
                    else:
                        raise
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.configs.append(config)
                self.metadata_list.append(metadata)
                
                log.info(f"모델 {i+1} 로딩 완료: {metadata.get('wandb_run_name', 'Unknown')} (device: {self.device})")
                
            except Exception as e:
                log.error(f"모델 {i+1} 로딩 실패: {e}")
                log.error(f"경로: {path}")
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
    
    def length_based_ensemble(self, input_texts, config):
        """
        길이 기반 앙상블: 각 모델의 결과 중 가장 긴 것을 선택
        
        Args:
            input_texts: 입력 텍스트 리스트
            config: 설정 딕셔너리
            
        Returns:
            list: 앙상블 결과 텍스트 리스트
        """
        results = []
        tokenizer = self.tokenizers[0]  # 기준 토크나이저
        
        log.info("길이 기반 앙상블 시작...")
        
        for text in tqdm(input_texts, desc="길이 기반 앙상블 처리 중"):
            try:
                # 입력 토크나이제이션
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 각 모델의 결과를 직접 생성하여 길이 기반 선택
                model_results = []
                for model in self.models:
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
                        
                        model_results.append(generated_text.strip())
                
                # 길이 기반 선택: 가장 긴 결과를 선택
                if model_results:
                    # 가장 긴 결과 선택
                    longest_result = max(model_results, key=len)
                    results.append(longest_result)
                else:
                    results.append("")
                    
            except Exception as e:
                log.warning(f"길이 기반 앙상블 중 오류 (fallback 사용): {e}")
                results.append("")  # 빈 문자열로 fallback
        
        log.info("길이 기반 앙상블 완료")
        return results
    
    def soft_voting_ensemble(self, input_texts, config):
        """
        진짜 소프트 보팅 앙상블: 모델들의 확률 분포를 평균하여 생성
        
        Args:
            input_texts: 입력 텍스트 리스트
            config: 설정 딕셔너리
            
        Returns:
            list: 앙상블 결과 텍스트 리스트
        """
        results = []
        tokenizer = self.tokenizers[0]  # 기준 토크나이저
        
        log.info("소프트 보팅 앙상블 시작...")
        
        for text in tqdm(input_texts, desc="소프트 보팅 앙상블 처리 중"):
            try:
                # 입력 토크나이제이션
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 각 모델에서 beam search를 통해 여러 후보 생성
                model_candidates = []
                for model in self.models:
                    with torch.no_grad():
                        # beam search로 여러 후보 생성
                        outputs = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                            num_return_sequences=min(3, config['inference']['num_beams']),  # 최대 3개 후보
                            return_dict_in_generate=True,
                            output_scores=True,
                            early_stopping=config['inference']['early_stopping']
                        )
                        
                        # 각 후보와 그 점수를 저장
                        candidates = []
                        for i, sequence in enumerate(outputs.sequences):
                            text_output = tokenizer.decode(sequence, skip_special_tokens=True)
                            
                            # 불필요한 토큰 제거
                            for token in config['inference']['remove_tokens']:
                                text_output = text_output.replace(token, " ")
                            
                            text_output = text_output.strip()
                            
                            # 점수 계산 (길이로 정규화된 평균 점수)
                            if hasattr(outputs, 'sequences_scores') and len(outputs.sequences_scores) > i:
                                score = outputs.sequences_scores[i].item()
                            else:
                                # sequences_scores가 없으면 길이 기반 점수 사용
                                score = len(text_output.split()) / config['inference']['generate_max_length']
                            
                            candidates.append((text_output, score))
                        
                        model_candidates.append(candidates)
                
                # 소프트 보팅: 각 모델의 최고 점수 후보들 중에서 평균 점수가 가장 높은 것 선택
                all_candidates = []
                
                # 각 모델의 모든 후보를 수집
                for model_idx, candidates in enumerate(model_candidates):
                    for text_output, score in candidates:
                        all_candidates.append((text_output, score, model_idx))
                
                if all_candidates:
                    # 동일한 텍스트에 대해 점수 평균 계산
                    text_scores = {}
                    text_counts = {}
                    
                    for text_output, score, model_idx in all_candidates:
                        if text_output not in text_scores:
                            text_scores[text_output] = 0
                            text_counts[text_output] = 0
                        text_scores[text_output] += score
                        text_counts[text_output] += 1
                    
                    # 평균 점수 계산
                    for text_output in text_scores:
                        text_scores[text_output] /= text_counts[text_output]
                    
                    # 가장 높은 평균 점수를 가진 텍스트 선택
                    best_text = max(text_scores.keys(), key=lambda x: text_scores[x])
                    results.append(best_text)
                else:
                    results.append("")
                    
            except Exception as e:
                log.warning(f"소프트 보팅 앙상블 중 오류 (fallback 사용): {e}")
                results.append("")  # 빈 문자열로 fallback
        
        log.info("소프트 보팅 앙상블 완료")
        return results
    
    def evaluate_on_validation(self, val_data_path):
        """
        검증 데이터로 앙상블 및 개별 모델 성능 평가
        
        Args:
            val_data_path: 검증 데이터 경로
            
        Returns:
            dict: 평가 결과 (개별 모델 점수, 앙상블 점수)
        """
        log.info(f"검증 데이터 평가 시작: {val_data_path}")
        
        # 검증 데이터 로드 (빠른 테스트를 위해 일부만 사용)
        val_df = pd.read_csv(val_data_path)
        # 빠른 테스트를 위해 처음 50개만 사용
        val_df = val_df.head(50)
        input_texts = val_df['dialogue'].tolist()
        reference_summaries = val_df['summary'].tolist()
        log.info(f"검증 데이터 로드 완룼: {len(input_texts)}개 샘플 (빠른 테스트용)")
        
        # 각 모델별로 독립적으로 생성
        all_generated_texts = []
        individual_scores = []
        
        for i, (model, tokenizer, config) in enumerate(zip(self.models, self.tokenizers, self.configs)):
            log.info(f"모델 {i+1}/{len(self.models)} 검증 점수 계산 시작 (baseline.py 방식)...")
            
            # baseline.py 방식으로 정확한 검증 점수 계산
            rouge_scores = evaluate_single_model_with_baseline(model, tokenizer, config)
            individual_scores.append({
                'model_index': i + 1,
                'model_metadata': self.metadata_list[i],
                'rouge_scores': rouge_scores
            })
            
            log.info(f"모델 {i+1} 검증 점수 (baseline.py 방식) - ROUGE-1: {rouge_scores['rouge-1']:.6f}, "
                    f"ROUGE-2: {rouge_scores['rouge-2']:.6f}, ROUGE-L: {rouge_scores['rouge-l']:.6f}")
            
            # 앙상블용 추론 데이터 준비 (빠른 테스트용)
            generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
            all_generated_texts.append(generated_texts)
        
        # 세 가지 앙상블 방식 모두 테스트
        log.info("\n=== 하드 보팅 vs 소프트 보팅 vs 길이 기반 비교 ===")
        
        # 1. 하드 보팅 앙상블
        log.info("하드 보팅 앙상블 시작...")
        hard_voting_results = self.token_level_hard_voting(all_generated_texts, self.tokenizers[0])
        
        # 2. 소프트 보팅 앙상블
        log.info("소프트 보팅 앙상블 시작...")
        soft_voting_results = self.soft_voting_ensemble(input_texts, self.configs[0])
        
        # 3. 길이 기반 앙상블
        log.info("길이 기반 앙상블 시작...")
        length_based_results = self.length_based_ensemble(input_texts, self.configs[0])
        
        # ROUGE 계산 함수 정의
        def calculate_rouge_scores(predictions, references, method_name):
            from rouge import Rouge
            rouge = Rouge()
            
            # 불필요한 토큰 제거
            cleaned_predictions = []
            cleaned_references = []
            for pred, ref in zip(predictions, references):
                pred_clean = pred.strip()
                ref_clean = ref.strip()
                for token in self.configs[0]['inference']['remove_tokens']:
                    pred_clean = pred_clean.replace(token, " ")
                    ref_clean = ref_clean.replace(token, " ")
                pred_clean = pred_clean.strip() if pred_clean.strip() else "empty"
                ref_clean = ref_clean.strip() if ref_clean.strip() else "empty"
                cleaned_predictions.append(pred_clean)
                cleaned_references.append(ref_clean)
            
            try:
                rouge_results = rouge.get_scores(cleaned_predictions, cleaned_references, avg=True)
                rouge_scores = {key: value["f"] for key, value in rouge_results.items()}
                # rouge-avg 계산 추가
                rouge_avg = (rouge_scores['rouge-1'] + rouge_scores['rouge-2'] + rouge_scores['rouge-l']) / 3
                rouge_scores['rouge-avg'] = rouge_avg
                
                log.info(f"{method_name} 검증 점수 - ROUGE-1: {rouge_scores['rouge-1']:.4f}, "
                        f"ROUGE-2: {rouge_scores['rouge-2']:.4f}, ROUGE-L: {rouge_scores['rouge-l']:.4f}, "
                        f"ROUGE-avg: {rouge_scores['rouge-avg']:.4f}")
                return rouge_scores
            except Exception as e:
                log.warning(f"{method_name} ROUGE 계산 오류: {e}")
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge-avg': 0.0}
        
        # 3. 세 방식의 ROUGE 점수 계산
        hard_voting_scores = calculate_rouge_scores(hard_voting_results, reference_summaries, "하드 보팅")
        soft_voting_scores = calculate_rouge_scores(soft_voting_results, reference_summaries, "소프트 보팅")
        length_based_scores = calculate_rouge_scores(length_based_results, reference_summaries, "길이 기반")
        
        # 4. 비교 결과 출력
        log.info("\n=== 앙상블 방식 비교 결과 ===")
        log.info(f"하드 보팅 ROUGE-avg: {hard_voting_scores['rouge-avg']:.4f}")
        log.info(f"소프트 보팅 ROUGE-avg: {soft_voting_scores['rouge-avg']:.4f}")
        log.info(f"길이 기반 ROUGE-avg: {length_based_scores['rouge-avg']:.4f}")
        
        # 가장 나은 방식 선택
        all_scores = {
            "하드 보팅": (hard_voting_scores, hard_voting_results),
            "소프트 보팅": (soft_voting_scores, soft_voting_results),
            "길이 기반": (length_based_scores, length_based_results)
        }
        
        best_method = max(all_scores.keys(), key=lambda x: all_scores[x][0]['rouge-avg'])
        ensemble_rouge_scores = all_scores[best_method][0]
        
        log.info(f"{best_method}이 가장 나은 성능을 보입니다!")
        
        evaluation_results = {
            'individual_model_scores': individual_scores,
            'hard_voting_scores': hard_voting_scores,
            'soft_voting_scores': soft_voting_scores,
            'length_based_scores': length_based_scores,
            'ensemble_scores': ensemble_rouge_scores,
            'best_ensemble_method': best_method,
            'num_validation_samples': len(input_texts)
        }
        
        log.info("검증 데이터 평가 완료 (baseline.py 방식 사용)")
        return evaluation_results
    
    def run_ensemble(self, test_data_path):
        """
        하드 보팅 앙상블 실행
        
        Args:
            test_data_path: 테스트 데이터 경로
            
        Returns:
            tuple: (ensemble_result_df, individual_results_list)
        """
        log.info(f"앙상블 추론 시작: {test_data_path}")
        
        # 테스트 데이터 로드 (빠른 테스트를 위해 일부만 사용)
        test_df = pd.read_csv(test_data_path)
        # 빠른 테스트를 위해 처음 20개만 사용
        test_df = test_df.head(20)
        input_texts = test_df['dialogue'].tolist()
        log.info(f"테스트 데이터 로드 완룼: {len(input_texts)}개 샘플 (빠른 테스트용)")
        
        # 개별 모델들로 추론 수행
        all_generated_texts = []
        
        for i, (model, tokenizer, config) in enumerate(zip(self.models, self.tokenizers, self.configs)):
            log.info(f"모델 {i+1}/{len(self.models)} 추론 시작...")
            log.info(f"모델 설정 - max_length: {config['inference']['generate_max_length']}, "
                    f"num_beams: {config['inference']['num_beams']}")
            
            generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
            all_generated_texts.append(generated_texts)
            
            log.info(f"모델 {i+1} 추론 완료")
        
        # 세 가지 앙상블 방식 모두 수행
        log.info("\n=== 하드 보팅 vs 소프트 보팅 vs 길이 기반 앙상블 수행 ===")
        
        # 1. 하드 보팅 앙상블
        log.info("하드 보팅 앙상블 시작...")
        hard_voting_results = self.token_level_hard_voting(all_generated_texts, self.tokenizers[0])
        
        # 2. 소프트 보팅 앙상블
        log.info("소프트 보팅 앙상블 시작...")
        soft_voting_results = self.soft_voting_ensemble(input_texts, self.configs[0])
        
        # 3. 길이 기반 앙상블
        log.info("길이 기반 앙상블 시작...")
        length_based_results = self.length_based_ensemble(input_texts, self.configs[0])
        
        # 4. 세 방식의 결과 데이터프레임 생성
        hard_voting_df = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': hard_voting_results
        })
        
        soft_voting_df = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': soft_voting_results
        })
        
        length_based_df = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': length_based_results
        })
        
        log.info("앙상블 추론 완료 (하드 보팅 & 소프트 보팅 & 길이 기반)")
        
        # 세 방식의 결과를 모두 반환
        ensemble_results = {
            'hard_voting': hard_voting_df,
            'soft_voting': soft_voting_df,
            'length_based': length_based_df,
            'individual_results': all_generated_texts
        }
        
        return ensemble_results, all_generated_texts

def run_single_method(method_name):
    """
    개별 앙상블 방식 실행 함수
    
    Args:
        method_name: 실행할 방식 ('hard_voting', 'soft_voting', 'length_based', 'realtime_token')
    """
    log.info(f"🎯 개별 방식 실행: {method_name}")
    
    # 공통 함수로 모델 경로 가져오기
    existing_model_paths = get_model_paths()
    if not existing_model_paths:
        return
    
    log.info(f"총 {len(existing_model_paths)}개 모델로 {method_name} 진행")
    
    # 디바이스 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 실시간 토큰 앙상블 방식
    if method_name == "realtime_token":
        ensemble = RealtimeTokenEnsemble(existing_model_paths, device=device)
        ensemble.load_models()
        
        # 검증 데이터 평가
        val_data_path = "../../input/data/dev.csv"
        if os.path.exists(val_data_path):
            log.info("검증 데이터 평가 시작")
            evaluation_results = ensemble.evaluate_on_validation(val_data_path)
            if evaluation_results:
                scores = evaluation_results['realtime_token_ensemble_scores']
                log.info(f"{method_name} 검증 점수 - ROUGE-avg: {scores['rouge-avg']:.4f}")
        
        # 테스트 데이터 추론
        test_data_path = "../../input/data/test.csv"
        if os.path.exists(test_data_path):
            ensemble_df, generation_time = ensemble.run_ensemble(test_data_path)
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "./ensemble_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_path = os.path.join(results_dir, f"{method_name}_{timestamp}.csv")
            ensemble_df.to_csv(result_path, index=False, encoding='utf-8')
            log.info(f"{method_name} 결과 저장: {result_path}")
            log.info(f"{method_name} 생성 시간: {generation_time:.2f}초")
    
    # 후처리 방식들 (hard_voting, soft_voting, length_based)
    else:
        ensemble = PostProcessingEnsemble(existing_model_paths, device=device)
        ensemble.load_models()
        
        # 검증 데이터로 개별 방식 평가
        val_data_path = "../../input/data/dev.csv"
        if os.path.exists(val_data_path):
            log.info("검증 데이터 평가 시작")
            val_df = pd.read_csv(val_data_path)
            val_df_sample = val_df.head(50)  # 빠른 테스트용
            input_texts = val_df_sample['dialogue'].tolist()
            reference_summaries = val_df_sample['summary'].tolist()
            
            # 선택한 방식으로만 생성
            if method_name == "hard_voting":
                # 모든 모델로 생성 후 하드 보팅
                generated_texts_list = []
                for model, tokenizer, config in zip(ensemble.models, ensemble.tokenizers, ensemble.configs):
                    texts = ensemble.generate_with_single_model(model, tokenizer, config, input_texts)
                    generated_texts_list.append(texts)
                results = ensemble.token_level_hard_voting(generated_texts_list, ensemble.tokenizers[0])
                
            elif method_name == "soft_voting":
                results = ensemble.soft_voting_ensemble(input_texts, ensemble.configs[0])
                
            elif method_name == "length_based":
                results = ensemble.length_based_ensemble(input_texts, ensemble.configs[0])
            
            # ROUGE 점수 계산
            from rouge import Rouge
            rouge = Rouge()
            cleaned_predictions = []
            cleaned_references = []
            for pred, ref in zip(results, reference_summaries):
                pred_clean = pred.strip() if pred.strip() else "empty"
                ref_clean = ref.strip() if ref.strip() else "empty"
                cleaned_predictions.append(pred_clean)
                cleaned_references.append(ref_clean)
            
            try:
                rouge_results = rouge.get_scores(cleaned_predictions, cleaned_references, avg=True)
                rouge_scores = {key: value["f"] for key, value in rouge_results.items()}
                rouge_avg = (rouge_scores['rouge-1'] + rouge_scores['rouge-2'] + rouge_scores['rouge-l']) / 3
                rouge_scores['rouge-avg'] = rouge_avg
                log.info(f"{method_name} 검증 점수 - ROUGE-avg: {rouge_scores['rouge-avg']:.4f}")
            except Exception as e:
                log.warning(f"ROUGE 계산 오류: {e}")
        
        # 테스트 데이터 추론
        test_data_path = "../../input/data/test.csv"
        if os.path.exists(test_data_path):
            test_df = pd.read_csv(test_data_path)
            test_df_sample = test_df.head(20)  # 빠른 테스트용
            test_input_texts = test_df_sample['dialogue'].tolist()
            
            # 선택한 방식으로만 생성
            if method_name == "hard_voting":
                generated_texts_list = []
                for model, tokenizer, config in zip(ensemble.models, ensemble.tokenizers, ensemble.configs):
                    texts = ensemble.generate_with_single_model(model, tokenizer, config, test_input_texts)
                    generated_texts_list.append(texts)
                final_results = ensemble.token_level_hard_voting(generated_texts_list, ensemble.tokenizers[0])
                
            elif method_name == "soft_voting":
                final_results = ensemble.soft_voting_ensemble(test_input_texts, ensemble.configs[0])
                
            elif method_name == "length_based":
                final_results = ensemble.length_based_ensemble(test_input_texts, ensemble.configs[0])
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "./ensemble_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_df = pd.DataFrame({
                'fname': test_df_sample['fname'],
                'summary': final_results
            })
            
            result_path = os.path.join(results_dir, f"{method_name}_{timestamp}.csv")
            result_df.to_csv(result_path, index=False, encoding='utf-8')
            log.info(f"{method_name} 결과 저장: {result_path}")
    
    log.info(f"🎉 {method_name} 실행 완료!")

def main(ensemble_strategy="comprehensive"):
    """
    앙상블 추론 메인 함수
    
    Args:
        ensemble_strategy: 앙상블 전략 ('comprehensive', 'hard_voting', 'soft_voting', 'length_based', 'realtime_token', 'post_token_voting', 'realtime_token_ensemble')
    """
    
    # 🔬 종합 실험 실행 (모든 방식 비교)
    if ensemble_strategy == "comprehensive":
        return main_comprehensive_experiment()
    
    # 🎯 개별 방식 실행
    if ensemble_strategy in ["hard_voting", "soft_voting", "length_based", "realtime_token"]:
        return run_single_method(ensemble_strategy)
    
    # 기존 단일 전략 실행 (하위 호환성)
    log.info(f"선택된 앙상블 전략: {ensemble_strategy}")
    
    # 공통 함수로 모델 경로 가져오기
    existing_model_paths = get_model_paths()
    if not existing_model_paths:
        return
    
    log.info(f"총 {len(existing_model_paths)}개 모델로 앙상블 진행")
    
    # 앙상블 객체 생성
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if ensemble_strategy == "realtime_token_ensemble":
        ensemble = RealtimeTokenEnsemble(existing_model_paths, device=device)
    else:  # post_token_voting (default)
        ensemble = PostProcessingEnsemble(existing_model_paths, device=device)
    
    # 모델들 로딩
    try:
        ensemble.load_models()
    except Exception as e:
        log.error(f"모델 로딩 실패: {e}")
        return
    
    # 검증 데이터로 성능 평가 실행
    val_data_path = "../../input/data/dev.csv"
    evaluation_results = None
    
    if os.path.exists(val_data_path):
        try:
            log.info("="*50)
            log.info("검증 데이터 성능 평가 시작")
            log.info("="*50)
            evaluation_results = ensemble.evaluate_on_validation(val_data_path)
            
            # 개별 모델 성능 로깅
            log.info("개별 모델 성능:")
            for score_info in evaluation_results['individual_model_scores']:
                model_idx = score_info['model_index']
                scores = score_info['rouge_scores']
                model_name = score_info['model_metadata'].get('wandb_run_name', f'Model_{model_idx}')
                log.info(f"  {model_name}: ROUGE-avg={scores['rouge-avg']:.4f}")
            
            # 앙상블 성능 로깅
            ensemble_scores = evaluation_results['ensemble_scores']
            log.info(f"앙상블 성능: ROUGE-avg={ensemble_scores['rouge-avg']:.4f}")
            
            # 개선 정도 계산
            best_individual_score = max([s['rouge_scores']['rouge-avg'] for s in evaluation_results['individual_model_scores']])
            improvement = ensemble_scores['rouge-avg'] - best_individual_score
            log.info(f"최고 개별 모델 대비 개선: {improvement:+.4f}")
            
        except Exception as e:
            log.error(f"검증 데이터 평가 실패: {e}")
    else:
        log.warning(f"검증 데이터 파일이 없습니다 (검증 점수 계산 건너뜨): {val_data_path}")
    
    # 앙상블 추론 실행
    test_data_path = "../../input/data/test.csv"
    if not os.path.exists(test_data_path):
        log.error(f"테스트 데이터 파일이 없습니다: {test_data_path}")
        return
    
    try:
        ensemble_results, individual_results = ensemble.run_ensemble(test_data_path)
    except Exception as e:
        log.error(f"앙상블 추론 실패: {e}")
        return
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ensemble_results 폴더 생성
    results_dir = "./ensemble_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 하드 보팅 결과 저장
    hard_voting_path = os.path.join(results_dir, f"ensemble_hard_voting_{timestamp}.csv")
    ensemble_results['hard_voting'].to_csv(hard_voting_path, index=False, encoding='utf-8')
    log.info(f"하드 보팅 앙상블 결과 저장: {hard_voting_path}")
    
    # 소프트 보팅 결과 저장
    soft_voting_path = os.path.join(results_dir, f"ensemble_soft_voting_{timestamp}.csv")
    ensemble_results['soft_voting'].to_csv(soft_voting_path, index=False, encoding='utf-8')
    log.info(f"소프트 보팅 앙상블 결과 저장: {soft_voting_path}")
    
    # 길이 기반 결과 저장
    length_based_path = os.path.join(results_dir, f"ensemble_length_based_{timestamp}.csv")
    ensemble_results['length_based'].to_csv(length_based_path, index=False, encoding='utf-8')
    log.info(f"길이 기반 앙상블 결과 저장: {length_based_path}")
    
    # 개별 모델 결과들 저장
    for i, individual_result in enumerate(individual_results):
        individual_df = pd.DataFrame({
            'fname': ensemble_results['hard_voting']['fname'],  # 하드 보팅 결과의 fname 사용
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
        "ensemble_strategies": ["hard_voting", "soft_voting", "length_based"],
        "model_metadata": ensemble.metadata_list,
        "evaluation_results": evaluation_results  # 검증 점수 결과 추가
    }
    
    metadata_path = os.path.join(results_dir, f"ensemble_comparison_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(ensemble_metadata, f, indent=2, ensure_ascii=False)
    log.info(f"앙상블 메타데이터 저장: {metadata_path}")
    
    log.info("=" * 50)
    log.info(f"앙상블 추론 완료! (하드 보팅 & 소프트 보팅 & 길이 기반)")
    log.info(f"사용된 모델 수: {len(existing_model_paths)}")
    log.info(f"하드 보팅 결과: {hard_voting_path}")
    log.info(f"소프트 보팅 결과: {soft_voting_path}")
    log.info(f"길이 기반 결과: {length_based_path}")
    
    # 검증 점수 요약 출력
    if evaluation_results:
        log.info(f"평가 결과 요약 (하드 vs 소프트 vs 길이 기반 비교):")
        
        # 하드 보팅 결과
        hard_scores = evaluation_results['hard_voting_scores']
        log.info(f"  하드 보팅 ROUGE-1: {hard_scores['rouge-1']:.4f}")
        log.info(f"  하드 보팅 ROUGE-2: {hard_scores['rouge-2']:.4f}")
        log.info(f"  하드 보팅 ROUGE-L: {hard_scores['rouge-l']:.4f}")
        log.info(f"  하드 보팅 ROUGE-avg: {hard_scores['rouge-avg']:.4f}")
        
        # 소프트 보팅 결과
        soft_scores = evaluation_results['soft_voting_scores']
        log.info(f"  소프트 보팅 ROUGE-1: {soft_scores['rouge-1']:.4f}")
        log.info(f"  소프트 보팅 ROUGE-2: {soft_scores['rouge-2']:.4f}")
        log.info(f"  소프트 보팅 ROUGE-L: {soft_scores['rouge-l']:.4f}")
        log.info(f"  소프트 보팅 ROUGE-avg: {soft_scores['rouge-avg']:.4f}")
        
        # 길이 기반 결과
        length_scores = evaluation_results['length_based_scores']
        log.info(f"  길이 기반 ROUGE-1: {length_scores['rouge-1']:.4f}")
        log.info(f"  길이 기반 ROUGE-2: {length_scores['rouge-2']:.4f}")
        log.info(f"  길이 기반 ROUGE-L: {length_scores['rouge-l']:.4f}")
        log.info(f"  길이 기반 ROUGE-avg: {length_scores['rouge-avg']:.4f}")
        
        # 최고 성능 방식
        best_method = evaluation_results.get('best_ensemble_method', 'Unknown')
        log.info(f"  최고 성능 방식: {best_method}")
        
        # 개별 모델 성능 비교
        best_individual_score = max([s['rouge_scores']['rouge-avg'] for s in evaluation_results['individual_model_scores']])
        hard_improvement = hard_scores['rouge-avg'] - best_individual_score
        soft_improvement = soft_scores['rouge-avg'] - best_individual_score
        length_improvement = length_scores['rouge-avg'] - best_individual_score
        log.info(f"  하드 보팅 개선: {hard_improvement:+.4f}")
        log.info(f"  소프트 보팅 개선: {soft_improvement:+.4f}")
        log.info(f"  길이 기반 개선: {length_improvement:+.4f}")
        
        # 개별 모델 성능 상세 정보
        log.info("개별 모델 성능 상세:")
        for i, score_info in enumerate(evaluation_results['individual_model_scores']):
            scores = score_info['rouge_scores']
            model_name = score_info['model_metadata'].get('wandb_run_name', f'Model_{i+1}')
            log.info(f"    {model_name}: ROUGE-avg={scores['rouge-avg']:.4f}")
    
    log.info("=" * 50)
    
    return evaluation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='앙상블 추론 시스템 - 여러 모델을 앙상블하여 텍스트 요약 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python ensemble_inference.py                     # 모든 방식 비교 실행
  python ensemble_inference.py --mode=all          # 모든 방식 비교 실행  
  python ensemble_inference.py --mode=hard_voting  # 하드 보팅만 실행
  python ensemble_inference.py --mode=soft_voting  # 소프트 보팅만 실행
  python ensemble_inference.py --mode=length_based # 길이 기반만 실행
  python ensemble_inference.py --mode=realtime_token # 실시간 토큰 앙상블만 실행

앙상블 방식 설명:
  all           - 4가지 방식을 모두 비교하여 최적 방식 추천
  hard_voting   - 각 모델이 완전한 텍스트 생성 후 토큰별 다수결
  soft_voting   - 각 모델의 확률 분포를 평균하여 최적 후보 선택
  length_based  - 각 모델 결과 중 가장 긴 것을 선택
  realtime_token- 매 토큰마다 모든 모델의 확률 분포를 평균하여 생성
        """)
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['all', 'hard_voting', 'soft_voting', 'length_based', 'realtime_token'],
        help='실행할 앙상블 방식 선택 (기본값: all - 모든 방식 비교)'
    )
    
    args = parser.parse_args()
    
    # 선택된 모드 로깅
    if args.mode == 'all':
        log.info("🔬 모든 앙상블 방식 비교 모드 시작")
        main("comprehensive")
    else:
        log.info(f"🎯 개별 방식 실행 모드: {args.mode}")
        main(args.mode)