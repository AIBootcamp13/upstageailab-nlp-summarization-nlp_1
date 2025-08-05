#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 사용방법
# 새 sweep 생성 후 5개 실험 실행
# python solar_api_sweep.py --count 5

# 기존 sweep ID로 추가 실험 실행
# python solar_api_sweep.py --sweep_id YOUR_SWEEP_ID --count 3

import sys
import yaml
import wandb
import pandas as pd
import time
from copy import deepcopy
from dotenv import load_dotenv
from tqdm import tqdm
from rouge import Rouge
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# solar_api.py에서 함수들 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_solar_client():
    """Solar API 클라이언트 설정"""
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    if not UPSTAGE_API_KEY:
        raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되어 있어야 합니다.")
    
    client = OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/solar"
    )
    return client

def load_data():
    """데이터셋 로드"""
    DATA_PATH = "../../input/data/"
    
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    return train_df, val_df, test_df

def compute_metrics(pred, gold):
    """ROUGE 점수 계산"""
    rouge = Rouge()
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

def build_prompt_fewshot2(dialogue, few_shot_dialogues, few_shot_summaries):
    """두 번째 퓨샷 방식 프롬프트 생성"""
    system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue."

    # 메시지 리스트 시작
    messages = [{"role": "system", "content": system_prompt}]
    
    # 여러 퓨샷 예제들을 assistant-user 대화 형태로 추가
    for i in range(len(few_shot_dialogues)):
        if i == 0:
            # 첫 번째 예제는 instruction 포함
            few_shot_user_prompt = (
                "Following the instructions below, summarize the given document.\n"
                "Instructions:\n"
                "1. Read the provided sample dialogue and corresponding summary.\n"
                "2. Read the dialogue carefully.\n"
                "3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\n\n"
                "Dialogue:\n"
                f"{few_shot_dialogues[i]}\n\n"
                "Summary:\n"
            )
        else:
            # 이후 예제들은 dialogue만
            few_shot_user_prompt = (
                "Dialogue:\n"
                f"{few_shot_dialogues[i]}\n\n"
                "Summary:\n"
            )
        
        messages.append({"role": "user", "content": few_shot_user_prompt})
        messages.append({"role": "assistant", "content": few_shot_summaries[i]})

    # 실제 질문 추가
    user_prompt = (
        "Dialogue:\n"
        f"{dialogue}\n\n"
        "Summary:\n"
    )
    messages.append({"role": "user", "content": user_prompt})

    return messages

def summarization_fewshot2(client, dialogue, few_shot_dialogues, few_shot_summaries, model, temperature, top_p):
    """두 번째 퓨샷 방식 요약 생성"""
    messages = build_prompt_fewshot2(dialogue, few_shot_dialogues, few_shot_summaries)
    
    summary = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return summary.choices[0].message.content

def validate_solar_api(client, train_df, val_df, model, temperature, top_p, few_shot_count, num_samples=100):
    """Solar API 검증 실행"""
    # Few-shot 샘플 생성
    few_shot_samples = train_df.sample(few_shot_count, random_state=42)
    
    few_shot_dialogues = []
    few_shot_summaries = []
    
    for i in range(len(few_shot_samples)):
        dialogue = few_shot_samples.iloc[i]['dialogue']
        summary = few_shot_samples.iloc[i]['summary']
        few_shot_dialogues.append(dialogue)
        few_shot_summaries.append(summary)
    
    # Validation 실행
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    
    scores = []
    all_predictions = []
    all_labels = []
    
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples), desc="Validation"):
        dialogue = row['dialogue']
        summary = summarization_fewshot2(
            client, dialogue, few_shot_dialogues, few_shot_summaries,
            model, temperature, top_p
        )
        results = compute_metrics(summary, row['summary'])
        avg_score = sum(results.values()) / len(results)

        scores.append(avg_score)
        all_predictions.append(summary)
        all_labels.append(row['summary'])
        
        # Rate limiting
        if (idx + 1) % 50 == 0:
            time.sleep(2)
    
    val_avg_score = sum(scores) / len(scores)
    
    # 상세한 ROUGE 분석
    remove_tokens = ['<usr>', '<s>', '</s>', '<pad>']
    replaced_predictions = all_predictions.copy()
    replaced_labels = all_labels.copy()
    
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

    rouge_evaluator = Rouge()
    results = rouge_evaluator.get_scores(replaced_predictions, replaced_labels, avg=True)
    
    # ROUGE 평균 계산
    f1_scores = [value["f"] for value in results.values()]
    rouge_avg = sum(f1_scores) / len(f1_scores)
    
    return val_avg_score, rouge_avg, results

def inference_solar_api(client, test_df, train_df, model, temperature, top_p, few_shot_count):
    """Solar API 추론 실행 및 CSV 생성"""
    # Few-shot 샘플 생성
    few_shot_samples = train_df.sample(few_shot_count, random_state=42)
    
    few_shot_dialogues = []
    few_shot_summaries = []
    
    for i in range(len(few_shot_samples)):
        dialogue = few_shot_samples.iloc[i]['dialogue']
        summary = few_shot_samples.iloc[i]['summary']
        few_shot_dialogues.append(dialogue)
        few_shot_summaries.append(summary)
    
    # 추론 실행
    summary_list = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        dialogue = row['dialogue']
        summary = summarization_fewshot2(
            client, dialogue, few_shot_dialogues, few_shot_summaries,
            model, temperature, top_p
        )
        summary_list.append(summary)

        # Rate limit 방지
        if (idx + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time

            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 5
                print(f"Waiting for {wait_time:.2f} sec for rate limiting")
                time.sleep(wait_time)

            start_time = time.time()
    
    # CSV 생성
    output = pd.DataFrame({
        "fname": test_df['fname'],
        "summary": summary_list,
    })
    
    # 결과 디렉토리 생성
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # 고유한 CSV 파일명 생성 (wandb run id 사용)
    run_id = wandb.run.id if wandb.run else "test"
    csv_filename = f"solar_api_sweep_{run_id}.csv"
    csv_path = os.path.join(result_path, csv_filename)
    
    output.to_csv(csv_path, index=False)
    
    return csv_path, output

def update_config_from_sweep(base_params, sweep_config):
    """WandB sweep에서 받은 하이퍼파라미터로 config 업데이트"""
    config = deepcopy(base_params)
    
    # wandb sweep 파라미터를 config에 반영
    for key, value in sweep_config.items():
        if key in config:
            config[key] = value
            print(f"Updated {key} = {value}")
    
    return config

def solar_api_sweep_train():
    """Solar API WandB sweep 실행을 위한 훈련 함수"""
    # wandb 초기화 (run_name에 solar_api_sweep 포함)
    wandb.init(name=f"solar_api_sweep_{wandb.util.generate_id()}")
    
    # sweep 설정 가져오기
    sweep_config = wandb.config
    
    # 기본 파라미터
    base_params = {
        "model": "solar-1-mini-chat",
        "temperature": 0.2,
        "top_p": 0.3,
        "few_shot_count": 1
    }
    
    # sweep 파라미터로 업데이트
    config = update_config_from_sweep(base_params, sweep_config)
    
    print("업데이트된 config 파라미터:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Solar API 클라이언트 설정
        client = setup_solar_client()
        
        # 데이터 로드
        train_df, val_df, test_df = load_data()
        
        # Validation 실행 (100 샘플 고정)
        print("Validation 시작...")
        val_avg_score, rouge_avg, rouge_results = validate_solar_api(
            client, train_df, val_df,
            config["model"], config["temperature"], config["top_p"], config["few_shot_count"],
            num_samples=100
        )
        
        # WandB에 메트릭 로깅
        wandb.log({
            "eval/rouge_avg": rouge_avg,
            "eval/rouge-1": rouge_results["rouge-1"]["f"],
            "eval/rouge-2": rouge_results["rouge-2"]["f"],
            "eval/rouge-l": rouge_results["rouge-l"]["f"],
            "val_avg_score": val_avg_score
        })
        
        print(f"Validation 완료. ROUGE Average: {rouge_avg:.6f}")
        
        # 추론 실행
        print("Test inference 시작...")
        csv_path, output_df = inference_solar_api(
            client, test_df, train_df,
            config["model"], config["temperature"], config["top_p"], config["few_shot_count"]
        )
        
        # WandB 아티팩트로 CSV 등록
        artifact = wandb.Artifact(
            name=f"solar_api_predictions_{wandb.run.id}",
            type="predictions",
            description=f"Solar API predictions with model={config['model']}, temp={config['temperature']}, few_shot={config['few_shot_count']}"
        )
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)
        
        print(f"추론 완료. CSV 저장: {csv_path}")
        print(f"WandB 아티팩트 등록 완료")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.finish(exit_code=1)
        raise
    
    # 정상 종료
    wandb.finish()

def create_sweep_from_yaml(yaml_path="config_sweep_solar.yaml", project_name="solar-api-sweep"):
    """YAML 파일에서 sweep 설정을 읽어와 sweep 생성"""
    # sweep 설정 읽기
    with open(yaml_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # sweep 생성
    sweep_id = wandb.sweep(
        sweep_config,
        project=project_name
    )
    
    return sweep_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar API WandB Sweep for Dialogue Summarization')
    parser.add_argument('--sweep_id', type=str, help='기존 sweep ID (없으면 새로 생성)')
    parser.add_argument('--sweep_config', type=str, default='config_sweep_solar.yaml', 
                        help='Sweep 설정 파일 경로')
    parser.add_argument('--count', type=int, default=1, 
                        help='실행할 sweep 실험 수')
    parser.add_argument('--project', type=str, default='solar-api-sweep',
                        help='WandB 프로젝트 이름')
    
    args = parser.parse_args()
    
    # sweep ID 결정
    if args.sweep_id is not None:
        sweep_id = args.sweep_id
        print(f"Using sweep ID from command line: {sweep_id}")
    else:
        # 새로운 sweep 생성
        print(f"Creating new sweep from {args.sweep_config}...")
        sweep_id = create_sweep_from_yaml(args.sweep_config, args.project)
        print(f"Created sweep with ID: {sweep_id}")
    
    # sweep agent 실행
    wandb.agent(
        sweep_id,
        function=solar_api_sweep_train,
        count=args.count
    )
    
    print(f"Completed {args.count} sweep runs!")