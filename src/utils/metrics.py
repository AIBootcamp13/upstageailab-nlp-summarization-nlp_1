"""
대회 공식 평가 지표(Metric) 계산을 위한 유틸리티 모음.

- Mecab 형태소 분석기 사용
- Multi-reference (다중 정답)에 대한 ROUGE 점수 평균 계산
"""

from typing import List, Dict
from konlpy.tag import Mecab
from rouge_score import rouge_scorer

# Mecab과 RougeScorer는 초기화에 시간이 걸릴 수 있으므로, 모듈 로드 시 한 번만 생성합니다.
try:
    _tagger = Mecab()
    _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=False)
except Exception as e:
    print(f"Mecab 또는 RougeScorer 초기화 실패: {e}")
    print("Mecab이 시스템에 설치되어 있는지 확인해주세요. `pip install konlpy mecab-ko mecab-ko-dic`")
    _tagger = None
    _scorer = None

def mecab_tokenize(text: str) -> str:
    """Mecab을 사용하여 텍스트를 형태소 단위로 토큰화합니다."""
    if not _tagger or not isinstance(text, str) or not text.strip():
        return ""
    
    # 특수 토큰 보호 (Person1, Person2 등)
    text = text.strip()
    
    # 불필요한 기호 제거 및 정규화
    import re
    text = re.sub(r'#Person\d+#:', '', text)  # Person 태그 제거
    text = re.sub(r'\s+', ' ', text)  # 다중 공백 정리
    
    morphs = _tagger.morphs(text)
    
    # 단일 문자 제거 (의미 없는 토큰)
    filtered_morphs = [m for m in morphs if len(m) > 1 or m in '.,!?']
    
    return " ".join(filtered_morphs)

def calculate_rouge_scores(preds: List[str], refs_list: List[List[str]]) -> Dict[str, float]:
    """
    대회 공식 규칙에 따라 ROUGE 점수를 계산합니다.

    - 각 예측(pred)에 대해, 모든 정답(refs)과의 ROUGE-F1 점수를 각각 계산한 후 평균을 냅니다.
    - 모든 예측에 대해 계산된 점수들의 최종 평균을 반환합니다.
    """
    if not _scorer or not preds or not refs_list:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0, "final_score": 0.0}

    total_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    valid_preds_count = 0

    for pred, refs in zip(preds, refs_list):
        if not pred or not any(r.strip() for r in refs):
            continue
        
        valid_preds_count += 1
        pred_tokenized = mecab_tokenize(pred)
        
        # 각 예측에 대한 정답들의 점수를 저장할 딕셔너리
        avg_ref_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
        valid_refs = [r for r in refs if r and r.strip()]
        
        if not valid_refs:
            valid_preds_count -= 1
            continue
        
        # 각 정답에 대해 점수를 계산하고 최대값 선택 (대회 공식 방식)
        ref_scores = {"rouge1": [], "rouge2": [], "rougeLsum": []}
        for ref in valid_refs:
            ref_tokenized = mecab_tokenize(ref)
            score = _scorer.score(ref_tokenized, pred_tokenized)
            for key in ref_scores:
                ref_scores[key].append(score[key].fmeasure)
        
        # 3개 정답 중 최대값 선택
        max_ref_scores = {key: max(scores) for key, scores in ref_scores.items()}
        
        # 최대값을 총합에 추가
        for key in total_scores:
            total_scores[key] += max_ref_scores[key]

    # 전체 예측 수만큼 나누어 최종 평균 점수 계산
    final_avg_scores = {key: val / valid_preds_count if valid_preds_count > 0 else 0.0 for key, val in total_scores.items()}
    
    # HuggingFace Trainer와 호환되도록 키 이름 변경 (e.g., rouge1 -> rouge_1)
    final_result = {
        'rouge_1': final_avg_scores['rouge1'],
        'rouge_2': final_avg_scores['rouge2'],
        'rouge_l': final_avg_scores['rougeLsum'],
    }
    
    # 대회 공식 규칙: 3개의 정답 요약 문장의 metric 최대값을 활용하여 최종 점수 계산
    # "최종 점수 = ((ROUGE-1 + ROUGE-2 + ROUGE-L) / 3) × 100"
    final_result['final_score'] = ((final_result['rouge_1'] + final_result['rouge_2'] + final_result['rouge_l']) / 3) * 100
    
    return final_result 