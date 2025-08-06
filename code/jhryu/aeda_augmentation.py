# -*- coding: utf-8 -*-
"""
AEDA (An Easier Data Augmentation) 기반 한국어 대화문 증강 시스템
API 없이 빠른 속도로 데이터 증강을 수행
"""

import os
import sys
import pandas as pd
import random
import re
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import json

# 스크립트 파일이 있는 디렉토리를 현재 작업 디렉토리로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../utils')
import log_util as log

class AEDAugmentator:
    def __init__(self, augmentation_strength='medium'):
        self.setup_korean_resources()
        self.setup_augmentation_params(augmentation_strength)
        self.setup_paths()
        
        # 증강 통계
        self.stats = {
            'total_processed': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'duplicate_filtered': 0,
            'processing_time': 0,
            'augmentation_methods': Counter()
        }
        
        log.info("AEDA 데이터 증강 시스템 초기화 완료")
        log.info(f"증강 강도: {augmentation_strength}")
        
    def setup_korean_resources(self):
        """한국어 특화 자원 설정"""
        
        # 한국어 감탄사 및 추임새
        self.korean_interjections = [
            '음', '아', '어', '그', '뭐', '아무튼', '그런데', '근데', '그러니까', '그니까',
            '아니', '어쨌든', '어찌됐든', '어쩌든', '그래서', '그럼', '그러면', '그치',
            '맞아', '맞다', '그래', '네', '예', '흠', '어머', '아이고', '이런', '저런',
            '하지만', '그러나', '그럼에도', '물론', '당연히', '확실히', '진짜', '정말',
            '아마', '혹시', '만약', '사실', '실제로', '솔직히', '사실은'
        ]
        
        # 자연스러운 구두점
        self.punctuations = [',', '...', '~', '-', '!']
        
        # 구두점별 삽입 확률
        self.punct_weights = {
            ',': 0.4,
            '...': 0.3,
            '~': 0.2,
            '-': 0.05,
            '!': 0.05
        }
        
        # 한국어 유의어 매핑 (기본적인 것들)
        self.synonym_dict = {
            # 긍정 표현
            '좋아': ['괜찮아', '좋네', '나쁘지 않아'],
            '좋아요': ['괜찮아요', '좋네요', '나쁘지 않아요', '좋습니다'],
            '괜찮아': ['좋아', '괜찮네', '문제없어'],
            '괜찮아요': ['좋아요', '괜찮네요', '문제없어요', '괜찮습니다'],
            
            # 부정 표현
            '안돼': ['안 돼', '힘들어', '어려워'],
            '안돼요': ['안 돼요', '힘들어요', '어려워요'],
            '나빠': ['별로야', '좋지 않아'],
            '나빠요': ['별로예요', '좋지 않아요'],
            
            # 응답 표현
            '네': ['예', '응', '맞아', '그래'],
            '네요': ['예요', '그렇네요', '맞네요'],
            '아니요': ['아뇨', '아니에요', '그게 아니에요'],
            
            # 일반 동사
            '가다': ['가기', '간다'],
            '오다': ['오기', '온다'],
            '먹다': ['먹기', '먹는다'],
            '하다': ['하기', '한다'],
            '말하다': ['말하기', '말한다', '얘기하다', '얘기한다'],
            
            # 시간 표현
            '지금': ['현재', '이때', '당장'],
            '나중에': ['이따가', '후에', '뒤에'],
            '빨리': ['빠르게', '서둘러', '급하게']
        }
        
        # 높임말/반말 변환
        self.formality_pairs = {
            '해요': '해',
            '가요': '가',
            '와요': '와',
            '좋아요': '좋아',
            '괜찮아요': '괜찮아',
            '맞아요': '맞아',
            '아니에요': '아니야',
            '그래요': '그래',
            '네요': '네',
            '예요': '야'
        }
        
    def setup_augmentation_params(self, strength):
        """증강 강도별 파라미터 설정"""
        if strength == 'light':
            self.aug_params = {
                'punctuation_prob': 0.1,
                'interjection_prob': 0.1,
                'synonym_prob': 0.05,
                'formality_prob': 0.05,
                'sentence_split_prob': 0.05,
                'max_changes_per_sentence': 1,
                'augmentation_per_sample': 1
            }
        elif strength == 'medium':
            self.aug_params = {
                'punctuation_prob': 0.2,
                'interjection_prob': 0.15,
                'synonym_prob': 0.1,
                'formality_prob': 0.1,
                'sentence_split_prob': 0.1,
                'max_changes_per_sentence': 2,
                'augmentation_per_sample': 2
            }
        elif strength == 'heavy':
            self.aug_params = {
                'punctuation_prob': 0.3,
                'interjection_prob': 0.2,
                'synonym_prob': 0.15,
                'formality_prob': 0.15,
                'sentence_split_prob': 0.15,
                'max_changes_per_sentence': 3,
                'augmentation_per_sample': 3
            }
    
    def setup_paths(self):
        """경로 설정"""
        self.data_path = "../../input/data/"
        self.output_path = "./augmented_data/"
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_data(self):
        """원본 데이터 로드"""
        train_path = os.path.join(self.data_path, 'train.csv')
        self.train_df = pd.read_csv(train_path)
        log.info(f"원본 데이터 로드 완료: {len(self.train_df)}개 샘플")
        return self.train_df
        
    def split_dialogue_sentences(self, dialogue):
        """대화를 문장별로 분리"""
        # #Person1#, #Person2# 기준으로 분리
        person_pattern = r'(#Person[12]#: [^#]*?)(?=#Person[12]#:|$)'
        sentences = re.findall(person_pattern, dialogue, re.DOTALL)
        return [s.strip() for s in sentences if s.strip()]
        
    def insert_punctuation(self, text):
        """구두점 삽입"""
        if random.random() > self.aug_params['punctuation_prob']:
            return text
            
        words = text.split()
        if len(words) < 3:
            return text
            
        # 랜덤한 위치에 구두점 삽입
        insert_pos = random.randint(1, len(words) - 1)
        punct = random.choices(
            list(self.punct_weights.keys()),
            weights=list(self.punct_weights.values())
        )[0]
        
        words.insert(insert_pos, punct)
        return ' '.join(words)
    
    def insert_interjection(self, sentence):
        """감탄사/추임새 삽입"""
        if random.random() > self.aug_params['interjection_prob']:
            return sentence
            
        if not sentence.startswith('#Person'):
            return sentence
            
        # #Person#: 다음에 감탄사 삽입
        pattern = r'(#Person[12]#: )(.*)'
        match = re.match(pattern, sentence)
        if match:
            prefix, content = match.groups()
            interjection = random.choice(self.korean_interjections)
            
            # 이미 감탄사로 시작하는 경우 스킵
            if content.strip().split()[0] in self.korean_interjections:
                return sentence
                
            new_content = f"{interjection}, {content}"
            return f"{prefix}{new_content}"
        
        return sentence
    
    def replace_synonyms(self, text):
        """유의어 교체"""
        if random.random() > self.aug_params['synonym_prob']:
            return text
            
        for word, synonyms in self.synonym_dict.items():
            if word in text:
                if random.random() < 0.3:  # 30% 확률로 교체
                    replacement = random.choice(synonyms)
                    text = text.replace(word, replacement, 1)  # 첫 번째만 교체
                    break
        
        return text
    
    def change_formality(self, text):
        """높임말/반말 변환"""
        if random.random() > self.aug_params['formality_prob']:
            return text
            
        # 높임말 → 반말 또는 그 반대
        for formal, informal in self.formality_pairs.items():
            if formal in text:
                if random.random() < 0.5:  # 50% 확률로 변환
                    text = text.replace(formal, informal, 1)
                    break
            elif informal in text:
                if random.random() < 0.5:
                    text = text.replace(informal, formal, 1)
                    break
        
        return text
    
    def split_or_merge_sentences(self, sentences):
        """문장 분할 또는 결합"""
        if random.random() > self.aug_params['sentence_split_prob']:
            return sentences
            
        if len(sentences) < 2:
            return sentences
            
        new_sentences = sentences.copy()
        
        # 랜덤하게 분할 또는 결합 선택
        if random.random() < 0.5 and len(sentences) > 2:
            # 문장 결합
            idx = random.randint(0, len(sentences) - 2)
            merged = f"{sentences[idx]} {sentences[idx + 1]}"
            new_sentences = sentences[:idx] + [merged] + sentences[idx + 2:]
        else:
            # 문장 분할 (긴 문장이 있는 경우)
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) > 10:  # 10단어 이상인 문장
                    words = sentence.split()
                    if len(words) > 6:
                        split_point = len(words) // 2
                        if sentence.startswith('#Person'):
                            # Person 태그 유지
                            pattern = r'(#Person[12]#: )(.*)'
                            match = re.match(pattern, sentence)
                            if match:
                                prefix, content = match.groups()
                                content_words = content.split()
                                if len(content_words) > 6:
                                    split_point = len(content_words) // 2
                                    part1 = f"{prefix}{' '.join(content_words[:split_point])}"
                                    part2 = f"{prefix}{' '.join(content_words[split_point:])}"
                                    new_sentences = sentences[:i] + [part1, part2] + sentences[i + 1:]
                                    break
        
        return new_sentences
    
    def augment_single_dialogue(self, dialogue, aug_method='all'):
        """단일 대화 증강"""
        try:
            # 대화를 문장별로 분리
            sentences = self.split_dialogue_sentences(dialogue)
            if not sentences:
                return None
                
            augmented_sentences = sentences.copy()
            changes_made = 0
            max_changes = self.aug_params['max_changes_per_sentence'] * len(sentences)
            
            # 각 문장에 대해 증강 기법 적용
            for i, sentence in enumerate(augmented_sentences):
                if changes_made >= max_changes:
                    break
                    
                original_sentence = sentence
                
                # 1. 구두점 삽입
                if aug_method in ['all', 'punctuation']:
                    sentence = self.insert_punctuation(sentence)
                    if sentence != original_sentence:
                        changes_made += 1
                        self.stats['augmentation_methods']['punctuation'] += 1
                
                # 2. 감탄사 삽입
                if aug_method in ['all', 'interjection']:
                    sentence = self.insert_interjection(sentence)
                    if sentence != original_sentence:
                        changes_made += 1
                        self.stats['augmentation_methods']['interjection'] += 1
                
                # 3. 유의어 교체
                if aug_method in ['all', 'synonym']:
                    sentence = self.replace_synonyms(sentence)
                    if sentence != original_sentence:
                        changes_made += 1
                        self.stats['augmentation_methods']['synonym'] += 1
                
                # 4. 높임말/반말 변환
                if aug_method in ['all', 'formality']:
                    sentence = self.change_formality(sentence)
                    if sentence != original_sentence:
                        changes_made += 1
                        self.stats['augmentation_methods']['formality'] += 1
                
                augmented_sentences[i] = sentence
            
            # 5. 문장 분할/결합
            if aug_method in ['all', 'sentence']:
                original_count = len(augmented_sentences)
                augmented_sentences = self.split_or_merge_sentences(augmented_sentences)
                if len(augmented_sentences) != original_count:
                    changes_made += 1
                    self.stats['augmentation_methods']['sentence'] += 1
            
            # 결과 재결합
            augmented_dialogue = '\n'.join(augmented_sentences)
            
            # 변화가 있었는지 확인
            if augmented_dialogue != dialogue and changes_made > 0:
                return augmented_dialogue.strip()
            else:
                return None
                
        except Exception as e:
            log.debug(f"증강 실패: {str(e)}")
            return None
    
    def augment_sample(self, row):
        """단일 샘플 증강 (여러 변형 생성)"""
        augmented_samples = []
        original_dialogue = row['dialogue']
        
        # 설정된 수만큼 증강 버전 생성
        aug_methods = ['all', 'punctuation', 'interjection', 'synonym', 'formality']
        
        for i in range(self.aug_params['augmentation_per_sample']):
            # 다양성을 위해 다른 증강 방법 사용
            method = random.choice(aug_methods)
            augmented_dialogue = self.augment_single_dialogue(original_dialogue, method)
            
            if augmented_dialogue and augmented_dialogue != original_dialogue:
                augmented_sample = {
                    'fname': f"{row['fname']}_aeda_{i+1}",
                    'dialogue': augmented_dialogue,
                    'summary': row['summary'],  # 요약은 원본 유지
                    'topic': row['topic'],
                    'augment_method': method,
                    'original_fname': row['fname']
                }
                augmented_samples.append(augmented_sample)
        
        return augmented_samples
    
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
    
    def run_augmentation(self, sample_size=None):
        """데이터 증강 실행"""
        log.info("="*60)
        log.info("AEDA 데이터 증강 시작")
        log.info("="*60)
        
        start_time = time.time()
        
        # 데이터 로드
        df = self.load_data()
        
        # 샘플 크기 제한 (테스트용)
        if sample_size:
            df = df.sample(min(sample_size, len(df))).reset_index(drop=True)
            log.info(f"테스트 모드: {len(df)}개 샘플로 제한")
        
        log.info(f"총 {len(df)}개 샘플 증강 시작")
        log.info(f"증강 강도: 샘플당 {self.aug_params['augmentation_per_sample']}개 변형 생성")
        
        # 진행바 설정
        augmented_data = []
        pbar = tqdm(total=len(df), desc="AEDA 증강 진행")
        
        for idx, row in df.iterrows():
            # 샘플 증강
            augmented_samples = self.augment_sample(row)
            
            if augmented_samples:
                augmented_data.extend(augmented_samples)
                self.stats['successful_augmentations'] += len(augmented_samples)
            else:
                self.stats['failed_augmentations'] += 1
            
            self.stats['total_processed'] += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'Success': self.stats['successful_augmentations'],
                'Failed': self.stats['failed_augmentations'],
                'Generated': len(augmented_data)
            })
        
        pbar.close()
        
        # 처리 시간 기록
        end_time = time.time()
        self.stats['processing_time'] = end_time - start_time
        
        log.info("AEDA 증강 완료!")
        log.info(f"총 처리: {self.stats['total_processed']}")
        log.info(f"성공한 증강: {self.stats['successful_augmentations']}")
        log.info(f"실패한 증강: {self.stats['failed_augmentations']}")
        log.info(f"처리 시간: {self.stats['processing_time']:.2f}초")
        log.info(f"처리 속도: {len(df) / self.stats['processing_time']:.1f}개/초")
        
        # 증강 기법별 통계
        if self.stats['augmentation_methods']:
            log.info("증강 기법별 적용 횟수:")
            for method, count in self.stats['augmentation_methods'].items():
                log.info(f"  {method}: {count}회")
        
        return augmented_data
    
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
        log.info(f"  - 원본 데이터: {len(original_data):,}개")
        log.info(f"  - 증강 데이터: {len(augmented_clean):,}개")
        log.info(f"  - 총 데이터: {len(final_df):,}개")
        log.info(f"  - 저장 위치: {output_file}")
        
        return output_file
    
    def generate_report(self, augmented_data, output_file):
        """품질 보고서 생성"""
        report = {
            "AEDA 데이터 증강 보고서": {
                "생성 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "원본 데이터 크기": len(self.train_df),
                "총 처리 샘플": self.stats['total_processed'],
                "성공한 증강": self.stats['successful_augmentations'],
                "실패한 증강": self.stats['failed_augmentations'],
                "중복 제거된 데이터": self.stats['duplicate_filtered'],
                "최종 증강 데이터": len(augmented_data),
                "최종 전체 데이터": len(self.train_df) + len(augmented_data),
                "처리 시간(초)": round(self.stats['processing_time'], 2),
                "처리 속도(개/초)": round(self.stats['total_processed'] / self.stats['processing_time'], 1)
            },
            "증강 기법별 적용 통계": dict(self.stats['augmentation_methods'])
        }
        
        # 보고서 저장
        report_file = os.path.join(self.output_path, 'aeda_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        log.info("="*60)
        log.info("AEDA 데이터 증강 완료 보고서")
        log.info("="*60)
        for section, data in report.items():
            log.info(f"{section}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    log.info(f"  {key}: {value}")
            else:
                log.info(f"  {data}")
            log.info("")

def main(test_mode=True, sample_size=10, strength='medium'):
    """메인 실행 함수"""
    augmentator = AEDAugmentator(augmentation_strength=strength)
    
    # 파라미터 확인
    if test_mode:
        log.info(f"테스트 모드: {sample_size}개 샘플로 시작")
    else:
        sample_size = None
        log.info("전체 데이터셋으로 증강을 시작합니다.")
    
    # 데이터 증강 실행
    augmented_data = augmentator.run_augmentation(sample_size=sample_size)
    
    if augmented_data:
        # 중복 제거 및 최종 데이터 정리
        clean_data = augmentator.remove_duplicates(augmented_data)
        
        # train2.csv 생성
        output_file = augmentator.create_train2_csv(clean_data)
        
        # 보고서 생성
        augmentator.generate_report(clean_data, output_file)
        
        log.info(f"모든 작업 완료! 결과 파일: {output_file}")
    else:
        log.error("증강된 데이터가 없습니다. 설정을 확인해주세요.")

if __name__ == "__main__":
    import sys
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            strength = sys.argv[2] if len(sys.argv) > 2 else 'medium'
            main(test_mode=False, strength=strength)
        elif sys.argv[1] == "--test":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            strength = sys.argv[3] if len(sys.argv) > 3 else 'medium'
            main(test_mode=True, sample_size=sample_size, strength=strength)
        else:
            print("사용법:")
            print("  python aeda_augmentation.py --test [샘플수] [강도]  # 테스트 모드")
            print("  python aeda_augmentation.py --full [강도]           # 전체 실행")
            print("  강도 옵션: light, medium, heavy")
    else:
        # 기본값: 테스트 모드
        main(test_mode=True, sample_size=10, strength='medium')