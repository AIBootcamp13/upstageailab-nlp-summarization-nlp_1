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
import random
import numpy as np

from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

# baseline.py에서 필요한 클래스들 임포트
from baseline import Preprocess, DatasetForVal, compute_metrics
from rouge import Rouge
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import tempfile

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
    
    def generate_ensemble_sequence(self, input_ids, attention_mask, config):
        """
        실시간 토큰 단위로 앙상블 시퀀스 생성
        
        Args:
            input_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크
            config: 생성 설정
            
        Returns:
            torch.Tensor: 생성된 시퀀스
        """
        batch_size = input_ids.size(0)
        max_length = config['inference']['generate_max_length']
        
        # 다음 토큰 예측을 위한 디코더 시작 토큰
        decoder_start_token_id = self.tokenizers[0].bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizers[0].eos_token_id
        
        # 초기 디코더 입력
        current_sequence = torch.full((batch_size, 1), decoder_start_token_id, 
                                    dtype=torch.long, device=self.device)
        
        # EOS 토큰 ID
        eos_token_id = self.tokenizers[0].eos_token_id
        
        # 완료된 시퀀스 추적
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(max_length - 1):  # -1 because we already have start token
            if finished.all():
                break
                
            # 각 모델에서 다음 토큰 logits 획득
            all_logits = []
            
            for model in self.models:
                with torch.no_grad():
                    # BART는 encoder-decoder 모델이므로 encoder output이 필요
                    encoder_outputs = model.get_encoder()(input_ids=input_ids, 
                                                         attention_mask=attention_mask)
                    
                    # 디코더로 다음 토큰 예측
                    decoder_outputs = model.get_decoder()(
                        input_ids=current_sequence,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=attention_mask
                    )
                    
                    # lm_head로 logits 계산
                    logits = model.lm_head(decoder_outputs.last_hidden_state)
                    
                    # 마지막 위치의 logits (다음 토큰 예측용)
                    next_token_logits = logits[:, -1, :]
                    all_logits.append(next_token_logits)
            
            # 균등 가중 평균으로 앙상블 (가중치 동일)
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            
            # greedy decoding (가장 확률이 높은 토큰 선택)
            next_tokens = torch.argmax(ensemble_logits, dim=-1)
            
            # EOS 토큰 체크 및 완료된 시퀀스 마킹
            finished = finished | (next_tokens == eos_token_id)
            
            # 완료되지 않은 시퀀스에만 토큰 추가
            next_tokens = next_tokens.unsqueeze(1)
            current_sequence = torch.cat([current_sequence, next_tokens], dim=1)
        
        return current_sequence
    
    def generate_with_realtime_ensemble(self, input_texts, config):
        """
        실시간 앙상블로 텍스트 생성
        
        Args:
            input_texts: 입력 텍스트 리스트
            config: 생성 설정
            
        Returns:
            list: 생성된 텍스트 리스트
        """
        results = []
        tokenizer = self.tokenizers[0]  # 기준 토큰나이저
        
        for text in tqdm(input_texts, desc="실시간 앙상블 생성 중"):
            try:
                # 입력 토큰화
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 실시간 앙상블 생성
                generated_sequence = self.generate_ensemble_sequence(
                    inputs['input_ids'], 
                    inputs['attention_mask'], 
                    config
                )
                
                # 텍스트로 디코딩
                generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                
                # 불필요한 토큰 제거
                for token in config['inference']['remove_tokens']:
                    generated_text = generated_text.replace(token, " ")
                
                results.append(generated_text.strip())
                
            except Exception as e:
                log.warning(f"실시간 앙상블 생성 중 오류 (fallback 사용): {e}")
                results.append("")  # 븈 문자열로 fallback
        
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
            if i == 0:  # 처음 모델에서만 데이터 준비
                generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
                all_generated_texts.append(generated_texts)
            else:
                generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
                all_generated_texts.append(generated_texts)
        
        # 하드 보팅으로 앙상블
        # 앙상블 전략에 따른 처리
        if isinstance(self, RealtimeTokenEnsemble):
            log.info("검증 데이터에 대한 실시간 토큰 앙상블 시작...")
            ensemble_results = self.generate_with_realtime_ensemble(input_texts, self.configs[0])
        else:
            log.info("검증 데이터에 대한 하드 보팅 앙상블 시작...")
            ensemble_results = self.token_level_hard_voting(all_generated_texts, self.tokenizers[0])
        
        # 앙상블 ROUGE 점수 계산 (기존 방식 사용, 샘플 데이터용)
        from rouge import Rouge
        rouge = Rouge()
        
        # 불필요한 토큰 제거
        cleaned_ensemble = []
        cleaned_references = []
        for pred, ref in zip(ensemble_results, reference_summaries):
            pred_clean = pred.strip()
            ref_clean = ref.strip()
            for token in self.configs[0]['inference']['remove_tokens']:
                pred_clean = pred_clean.replace(token, " ")
                ref_clean = ref_clean.replace(token, " ")
            pred_clean = pred_clean.strip() if pred_clean.strip() else "empty"
            ref_clean = ref_clean.strip() if ref_clean.strip() else "empty"
            cleaned_ensemble.append(pred_clean)
            cleaned_references.append(ref_clean)
        
        try:
            ensemble_rouge_results = rouge.get_scores(cleaned_ensemble, cleaned_references, avg=True)
            ensemble_rouge_scores = {key: value["f"] for key, value in ensemble_rouge_results.items()}
            # rouge-avg 계산 추가
            rouge_avg = (ensemble_rouge_scores['rouge-1'] + ensemble_rouge_scores['rouge-2'] + ensemble_rouge_scores['rouge-l']) / 3
            ensemble_rouge_scores['rouge-avg'] = rouge_avg
        except Exception as e:
            log.warning(f"앙상블 ROUGE 계산 오류: {e}")
            ensemble_rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge-avg': 0.0}
        
        log.info(f"앙상블 검증 점수 (샘플 데이터) - ROUGE-1: {ensemble_rouge_scores['rouge-1']:.4f}, "
                f"ROUGE-2: {ensemble_rouge_scores['rouge-2']:.4f}, ROUGE-L: {ensemble_rouge_scores['rouge-l']:.4f}")
        
        evaluation_results = {
            'individual_model_scores': individual_scores,
            'ensemble_scores': ensemble_rouge_scores,
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
        
        # 앙상블 전략에 따른 처리
        if isinstance(self, RealtimeTokenEnsemble):
            log.info("실시간 토큰 앙상블로 추론 수행...")
            ensemble_results = self.generate_with_realtime_ensemble(input_texts, self.configs[0])
            # 개별 모델 결과는 비교를 위해 생성
            all_generated_texts = []
            for i, (model, tokenizer, config) in enumerate(zip(self.models, self.tokenizers, self.configs)):
                log.info(f"비교용 모델 {i+1}/{len(self.models)} 추론 시작...")
                generated_texts = self.generate_with_single_model(model, tokenizer, config, input_texts)
                all_generated_texts.append(generated_texts)
        else:
            # 기존 PostTokenVoting 방식
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

def main(ensemble_strategy="post_token_voting"):
    """
    앙상블 추론 메인 함수
    
    Args:
        ensemble_strategy: 앙상블 전략 ('post_token_voting' 또는 'realtime_token_ensemble')
    """
    
    # 사용할 모델 경로들 (수동 지정)
    # TODO: 실제 저장된 모델 경로로 수정 필요
    model_paths = [
        "./models/model_baseline_20250804_063540.zip",  
        "./models/model_baseline_20250804_064025.zip",
    ]
    
    log.info(f"선택된 앙상블 전략: {ensemble_strategy}")
    
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
    
    if ensemble_strategy == "realtime_token_ensemble":
        ensemble = RealtimeTokenEnsemble(existing_model_paths, device=device)
    else:  # post_token_voting (default)
        ensemble = HardVotingEnsemble(existing_model_paths, device=device)
    
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
        result_df, individual_results = ensemble.run_ensemble(test_data_path)
    except Exception as e:
        log.error(f"앙상블 추론 실패: {e}")
        return
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ensemble_results 폴더 생성
    results_dir = "./ensemble_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 앙상블 결과 저장 (전략명 포함)
    strategy_name = "realtime_token_ensemble" if isinstance(ensemble, RealtimeTokenEnsemble) else "post_token_voting"
    ensemble_path = os.path.join(results_dir, f"ensemble_{strategy_name}_{timestamp}.csv")
    result_df.to_csv(ensemble_path, index=False, encoding='utf-8')
    log.info(f"앙상블 결과 저장: {ensemble_path}")
    
    # 개별 모델 결과들 저장 (실제 처리된 데이터와 길이 맞춤)
    # result_df에 이미 사용된 fname을 재사용
    for i, individual_result in enumerate(individual_results):
        individual_df = pd.DataFrame({
            'fname': result_df['fname'],  # 이미 처리된 데이터의 fname 사용
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
        "ensemble_strategy": strategy_name,
        "model_metadata": ensemble.metadata_list,
        "evaluation_results": evaluation_results  # 검증 점수 결과 추가
    }
    
    metadata_path = os.path.join(results_dir, f"ensemble_{strategy_name}_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(ensemble_metadata, f, indent=2, ensure_ascii=False)
    log.info(f"앙상블 메타데이터 저장: {metadata_path}")
    
    log.info("=" * 50)
    log.info(f"앙상블 추론 완료! (전략: {strategy_name})")
    log.info(f"사용된 모델 수: {len(existing_model_paths)}")
    log.info(f"앙상블 결과: {ensemble_path}")
    
    # 검증 점수 요약 출력
    if evaluation_results:
        log.info(f"평가 결과 요약 ({strategy_name}):")
        ensemble_scores = evaluation_results['ensemble_scores']
        log.info(f"  앙상블 ROUGE-1: {ensemble_scores['rouge-1']:.4f}")
        log.info(f"  앙상블 ROUGE-2: {ensemble_scores['rouge-2']:.4f}")
        log.info(f"  앙상블 ROUGE-L: {ensemble_scores['rouge-l']:.4f}")
        log.info(f"  앙상블 ROUGE-avg: {ensemble_scores['rouge-avg']:.4f}")
        
        best_individual_score = max([s['rouge_scores']['rouge-avg'] for s in evaluation_results['individual_model_scores']])
        improvement = ensemble_scores['rouge-avg'] - best_individual_score
        log.info(f"  최고 개별 모델 대비 개선: {improvement:+.4f}")
        
        # 개별 모델 성능 상세 정보
        log.info("개별 모델 성능 상세:")
        for i, score_info in enumerate(evaluation_results['individual_model_scores']):
            scores = score_info['rouge_scores']
            model_name = score_info['model_metadata'].get('wandb_run_name', f'Model_{i+1}')
            log.info(f"    {model_name}: ROUGE-avg={scores['rouge-avg']:.4f}")
    
    log.info("=" * 50)
    
    return evaluation_results

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 앙상블 전략 선택 가능
    ensemble_strategy = "post_token_voting"  # 기본값
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["realtime", "realtime_token_ensemble"]:
            ensemble_strategy = "realtime_token_ensemble"
        elif sys.argv[1] in ["post", "post_token_voting"]:
            ensemble_strategy = "post_token_voting"
        elif sys.argv[1] in ["both", "compare"]:
            # 두 전략 모두 실행
            log.info("\n" + "="*60)
            log.info("두 앙상블 전략 비교 실행")
            log.info("\n1. Post Token Voting 전략 실행")
            log.info("="*60)
            main("post_token_voting")
            
            log.info("\n" + "="*60)
            log.info("\n2. Realtime Token Ensemble 전략 실행")
            log.info("="*60)
            main("realtime_token_ensemble")
            exit()
    
    main(ensemble_strategy)