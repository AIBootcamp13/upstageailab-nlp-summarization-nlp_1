# -*- coding: utf-8 -*-
"""
NLP 대화문 요약 데이터 증강 스크립트
Solar API를 사용하여 기존 train.csv를 증강하여 train2.csv 생성
"""

import os
import sys
import pandas as pd
import time
import json
import random
from datetime import datetime
from tqdm import tqdm
from rouge import Rouge
from openai import OpenAI
from dotenv import load_dotenv

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../utils')
import log_util as log

class DataAugmentator:
    def __init__(self):
        self.setup_environment()
        self.setup_api()
        self.setup_paths()
        self.rouge = Rouge()
        
        # 증강 통계
        self.stats = {
            'total_processed': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'rouge_scores': []
        }
        
    def setup_environment(self):
        """환경 설정"""
        load_dotenv()
        
        # API 파라미터 설정
        self.params = {
            "model": "solar-1-mini-chat",
            "temperature": 0.3,  # 약간의 창의성 허용
            "top_p": 0.7,
            "max_tokens": 1000
        }
        
        log.info("데이터 증강 시스템 초기화 완료")
        for key, value in self.params.items():
            log.info(f"  {key}: {value}")
    
    def setup_api(self):
        """Solar API 클라이언트 설정"""
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        
    def setup_paths(self):
        """경로 설정"""
        self.data_path = "../../input/data/"
        self.output_path = "./augmented_data/"
        self.progress_path = "./progress/"
        
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.progress_path, exist_ok=True)
        
    def load_data(self):
        """원본 데이터 로드"""
        train_path = os.path.join(self.data_path, 'train.csv')
        self.train_df = pd.read_csv(train_path)
        log.info(f"원본 데이터 로드 완료: {len(self.train_df)}개 샘플")
        
        # 샘플 데이터 확인
        log.info("데이터 샘플:")
        log.info(f"dialogue 예시: {self.train_df.iloc[0]['dialogue'][:100]}...")
        log.info(f"summary 예시: {self.train_df.iloc[0]['summary']}")
        
    def get_augmentation_prompt(self, augment_type, dialogue, summary):
        """증강 유형별 프롬프트 생성"""
        
        if augment_type == "paraphrase":
            system_prompt = (
                "당신은 대화문 패러프레이징 전문가입니다. "
                "원본 대화의 핵심 의미와 내용을 정확히 유지하면서, "
                "표현 방식을 자연스럽게 변경해주세요."
            )
            user_prompt = f"""
다음 대화를 패러프레이징해주세요. 
원본의 의미를 그대로 유지하되, 다음과 같은 변경을 적용해주세요:
- 존댓말/반말 변환
- 유사한 의미의 다른 표현 사용
- 자연스러운 어조 변경

원본 대화:
{dialogue}

원본 요약:
{summary}

패러프레이징된 대화:
"""

        elif augment_type == "expand":
            system_prompt = (
                "당신은 대화문 확장 전문가입니다. "
                "원본 대화의 맥락을 유지하면서 자연스러운 대화 요소를 추가하여 "
                "더 풍부한 대화로 만들어주세요."
            )
            user_prompt = f"""
다음 대화를 자연스럽게 확장해주세요:
- 인사말, 감탄사, 추임새 등 자연스러운 요소 추가
- 대화의 흐름과 맥락에 맞는 부연설명 추가
- 원본의 핵심 내용과 결론은 반드시 유지

원본 대화:
{dialogue}

원본 요약:
{summary}

확장된 대화:
"""

        elif augment_type == "summary_restyle":
            system_prompt = (
                "당신은 요약 전문가입니다. "
                "주어진 대화는 그대로 유지하고, 요약만 다른 스타일로 재작성해주세요."
            )
            user_prompt = f"""
다음 대화에 대해 새로운 스타일의 요약을 작성해주세요:
- 핵심 정보는 동일하게 유지
- 표현 방식만 변경 (간결형↔상세형, 설명형↔개조식 등)
- 대화의 주요 내용을 빠뜨리지 않고 포함

대화:
{dialogue}

기존 요약:
{summary}

새로운 스타일 요약:
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def call_solar_api(self, messages):
        """Solar API 호출"""
        try:
            api_params = {k: v for k, v in self.params.items() if v is not None}
            api_params["messages"] = messages
            
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            log.error(f"API 호출 실패: {str(e)}")
            return None
    
    def calculate_rouge_score(self, pred, gold):
        """ROUGE 점수 계산"""
        try:
            # 토큰 정리
            remove_tokens = ['<usr>', '<s>', '</s>', '<pad>']
            for token in remove_tokens:
                pred = pred.replace(token, " ")
                gold = gold.replace(token, " ")
            
            results = self.rouge.get_scores(pred, gold, avg=True)
            return results['rouge-l']['f']
        except:
            return 0.0
    
    def augment_sample(self, row, augment_type):
        """단일 샘플 증강"""
        dialogue = row['dialogue']
        summary = row['summary']
        
        # 프롬프트 생성
        messages = self.get_augmentation_prompt(augment_type, dialogue, summary)
        
        # API 호출
        result = self.call_solar_api(messages)
        if not result:
            return None
            
        # 결과 파싱
        if augment_type == "summary_restyle":
            # 요약 재작성의 경우 대화는 원본 유지
            augmented_dialogue = dialogue
            augmented_summary = result
        else:
            # 대화 변경의 경우
            augmented_dialogue = result
            augmented_summary = summary
        
        # 품질 검증
        rouge_score = self.calculate_rouge_score(augmented_summary, summary)
        
        return {
            'fname': f"{row['fname']}_aug_{augment_type}",
            'dialogue': augmented_dialogue,
            'summary': augmented_summary,
            'topic': row['topic'],
            'augment_type': augment_type,
            'rouge_score': rouge_score,
            'original_fname': row['fname']
        }
    
    def save_progress(self, processed_data, current_index):
        """진행상황 저장"""
        progress_file = os.path.join(self.progress_path, 'augmentation_progress.json')
        
        progress_data = {
            'current_index': current_index,
            'processed_count': len(processed_data),
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
        # 중간 결과 저장
        if processed_data:
            temp_df = pd.DataFrame(processed_data)
            temp_file = os.path.join(self.progress_path, 'temp_augmented_data.csv')
            temp_df.to_csv(temp_file, index=False, encoding='utf-8')
            
    def load_progress(self):
        """진행상황 로드"""
        progress_file = os.path.join(self.progress_path, 'augmentation_progress.json')
        temp_file = os.path.join(self.progress_path, 'temp_augmented_data.csv')
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                
            processed_data = []
            if os.path.exists(temp_file):
                temp_df = pd.read_csv(temp_file)
                processed_data = temp_df.to_dict('records')
                
            log.info(f"이전 진행상황 로드: {progress_data['current_index']}번째부터 재시작")
            return progress_data['current_index'], processed_data
            
        return 0, []
    
    def run_augmentation(self, sample_size=None, min_rouge_score=0.3):
        """데이터 증강 실행"""
        log.info("="*60)
        log.info("데이터 증강 시작")
        log.info("="*60)
        
        # 데이터 로드
        self.load_data()
        
        # 이전 진행상황 확인
        start_index, processed_data = self.load_progress()
        
        # 샘플 크기 설정 (테스트용)
        if sample_size:
            df = self.train_df.sample(sample_size).reset_index(drop=True)
            log.info(f"테스트 모드: {sample_size}개 샘플로 제한")
        else:
            df = self.train_df
            
        # 증강 유형 분배
        augment_types = ['paraphrase', 'expand', 'summary_restyle']
        total_samples = len(df)
        
        log.info(f"총 {total_samples}개 샘플 증강 예정")
        log.info(f"증강 유형별 분배: {len(augment_types)}가지 유형")
        
        # 진행바 설정
        pbar = tqdm(total=total_samples, initial=start_index, 
                   desc="데이터 증강 진행")
        
        batch_start_time = time.time()
        api_call_count = 0
        
        for idx in range(start_index, total_samples):
            row = df.iloc[idx]
            
            # 증강 유형 선택 (순환)
            augment_type = augment_types[idx % len(augment_types)]
            
            # 샘플 증강
            augmented_sample = self.augment_sample(row, augment_type)
            
            if augmented_sample:
                # 품질 필터링
                if augmented_sample['rouge_score'] >= min_rouge_score:
                    processed_data.append(augmented_sample)
                    self.stats['successful_augmentations'] += 1
                    self.stats['rouge_scores'].append(augmented_sample['rouge_score'])
                else:
                    self.stats['quality_filtered'] += 1
                    log.debug(f"품질 기준 미달로 제외: ROUGE-L {augmented_sample['rouge_score']:.3f}")
            else:
                self.stats['failed_augmentations'] += 1
                
            self.stats['total_processed'] += 1
            api_call_count += 1
            
            # Rate limit 관리 (100개마다 1분 대기)
            if api_call_count % 100 == 0:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time + 5
                    log.info(f"Rate limit 관리: {wait_time:.1f}초 대기")
                    time.sleep(wait_time)
                
                batch_start_time = time.time()
                
            # 진행상황 저장 (50개마다)
            if (idx + 1) % 50 == 0:
                self.save_progress(processed_data, idx + 1)
                
            pbar.update(1)
            pbar.set_postfix({
                'Success': self.stats['successful_augmentations'],
                'Failed': self.stats['failed_augmentations'],
                'Filtered': self.stats['quality_filtered']
            })
            
        pbar.close()
        
        log.info("데이터 증강 완료!")
        log.info(f"총 처리: {self.stats['total_processed']}")
        log.info(f"성공: {self.stats['successful_augmentations']}")
        log.info(f"실패: {self.stats['failed_augmentations']}")
        log.info(f"품질 필터링: {self.stats['quality_filtered']}")
        
        if self.stats['rouge_scores']:
            avg_rouge = sum(self.stats['rouge_scores']) / len(self.stats['rouge_scores'])
            log.info(f"평균 ROUGE-L 점수: {avg_rouge:.4f}")
        
        return processed_data
    
    def remove_duplicates(self, augmented_data):
        """중복 데이터 제거"""
        log.info("중복 데이터 제거 중...")
        
        # DataFrame 생성
        df = pd.DataFrame(augmented_data)
        
        # 대화 내용 기준 중복 제거
        before_count = len(df)
        df_dedup = df.drop_duplicates(subset=['dialogue'], keep='first')
        after_count = len(df_dedup)
        
        removed_count = before_count - after_count
        self.stats['duplicate_filtered'] = removed_count
        
        log.info(f"중복 제거 완료: {removed_count}개 중복 데이터 제거")
        
        return df_dedup.to_dict('records')
    
    def create_train2_csv(self, augmented_data):
        """최종 train2.csv 생성"""
        log.info("최종 train2.csv 파일 생성 중...")
        
        # 중복 제거
        clean_data = self.remove_duplicates(augmented_data)
        
        # 원본 데이터와 증강 데이터 결합
        original_data = self.train_df[['fname', 'dialogue', 'summary', 'topic']].to_dict('records')
        
        # 증강 데이터에서 필요한 컬럼만 추출
        augmented_clean = []
        for item in clean_data:
            augmented_clean.append({
                'fname': item['fname'],
                'dialogue': item['dialogue'],
                'summary': item['summary'],
                'topic': item['topic']
            })
        
        # 결합
        combined_data = original_data + augmented_clean
        
        # DataFrame 생성 및 저장
        final_df = pd.DataFrame(combined_data)
        output_file = os.path.join(self.output_path, 'train2.csv')
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        
        log.info(f"train2.csv 생성 완료!")
        log.info(f"  - 원본 데이터: {len(original_data)}개")
        log.info(f"  - 증강 데이터: {len(augmented_clean)}개")
        log.info(f"  - 총 데이터: {len(final_df)}개")
        log.info(f"  - 저장 위치: {output_file}")
        
        return output_file
    
    def generate_report(self, augmented_data, output_file):
        """품질 보고서 생성"""
        report = {
            "데이터 증강 보고서": {
                "생성 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "원본 데이터 크기": len(self.train_df),
                "총 처리 샘플": self.stats['total_processed'],
                "성공한 증강": self.stats['successful_augmentations'],
                "실패한 증강": self.stats['failed_augmentations'],
                "품질 필터링된 데이터": self.stats['quality_filtered'],
                "중복 제거된 데이터": self.stats['duplicate_filtered'],
                "최종 증강 데이터": len(augmented_data),
                "최종 전체 데이터": len(self.train_df) + len(augmented_data)
            }
        }
        
        if self.stats['rouge_scores']:
            report["ROUGE 점수 통계"] = {
                "평균": sum(self.stats['rouge_scores']) / len(self.stats['rouge_scores']),
                "최소": min(self.stats['rouge_scores']),
                "최대": max(self.stats['rouge_scores']),
                "샘플 수": len(self.stats['rouge_scores'])
            }
        
        # 보고서 저장
        report_file = os.path.join(self.output_path, 'augmentation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        log.info("="*60)
        log.info("데이터 증강 완료 보고서")
        log.info("="*60)
        for section, data in report.items():
            log.info(f"{section}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    log.info(f"  {key}: {value}")
            else:
                log.info(f"  {data}")
            log.info("")

def main(test_mode=True, sample_size=10, min_rouge=0.3):
    """메인 실행 함수"""
    augmentator = DataAugmentator()
    
    # 파라미터 확인
    if test_mode:
        log.info(f"테스트 모드: {sample_size}개 샘플로 시작")
    else:
        sample_size = None
        log.info("전체 데이터셋으로 증강을 시작합니다.")
    
    # 데이터 증강 실행
    augmented_data = augmentator.run_augmentation(
        sample_size=sample_size, 
        min_rouge_score=min_rouge
    )
    
    if augmented_data:
        # train2.csv 생성
        output_file = augmentator.create_train2_csv(augmented_data)
        
        # 보고서 생성
        augmentator.generate_report(augmented_data, output_file)
        
        log.info(f"모든 작업 완료! 결과 파일: {output_file}")
    else:
        log.error("증강된 데이터가 없습니다. 설정을 확인해주세요.")

if __name__ == "__main__":
    import sys
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            main(test_mode=False, min_rouge=0.3)
        elif sys.argv[1] == "--test":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            main(test_mode=True, sample_size=sample_size, min_rouge=0.3)
        else:
            print("사용법:")
            print("  python data_augmentation.py --test [샘플수]  # 테스트 모드")
            print("  python data_augmentation.py --full          # 전체 실행")
    else:
        # 기본값: 테스트 모드
        main(test_mode=True, sample_size=5, min_rouge=0.3)