from __future__ import annotations

"""공통 후처리 유틸리티.

train.py 의 compute_metrics 와 evaluate.py 모두에서 동일한 규칙으로
텍스트 후처리를 수행하기 위해 만든 모듈이다.
"""

from typing import List
import re

# 기본적으로 제거할 토큰들 – config.yaml 의 inference.remove_tokens 와 동일하게 유지
DEFAULT_REMOVE_TOKENS: List[str] = [
    "<usr>",
    "<s>",
    "</s>",
    "<pad>",
]


def postprocess(text: str, remove_tokens: List[str] | None = None) -> str:
    """고도화된 후처리 (이전 성능 좋았던 버전 기반)
    
    1. remove_tokens 제거
    2. Person 태그 변환 및 복원
    3. 중복 제거 및 문장 품질 개선
    4. 길이 최적화
    """
    if not isinstance(text, str):
        return ""

    # 1) 특수 토큰 제거
    if remove_tokens is None:
        remove_tokens = DEFAULT_REMOVE_TOKENS
    for token in remove_tokens:
        text = text.replace(token, " ")

    # 2) Person 태그 변환
    text = re.sub(r"\b[Pp]erson\s*(\d+)\b", r"#Person\1#", text)

    # 3) 휴리스틱 기반 태그 복원
    text = re.sub(r"^는\s+", "#Person2#는 ", text)
    text = re.sub(r"\s+는\s+", " #Person2#는 ", text)
    text = re.sub(r"^과\s+", "#Person1#과 ", text)
    text = re.sub(r"\s+과\s+", " #Person1#과 ", text)
    text = re.sub(r"^이\s+", "#Person1#이 ", text)
    text = re.sub(r"\s+이\s+", " #Person1#이 ", text)
    text = re.sub(r"\s+에게\s+", " #Person1#에게 ", text)

    # 4) 연속 중복 태그 제거 1차 (태그 보완 전에 먼저 실행)
    text = re.sub(r"(#Person\d#)(?:\s*\1)+", r"\1", text)

    # 5) 조사 앞 누락 태그 보완 (은/는/이/가/과/와/에게/한테)
    tag_insert_rules = {
        r"\b은\s": "#Person2#은 ",
        r"\b는\s": "#Person2#는 ",
        r"\b이\s": "#Person1#이 ",
        r"\b가\s": "#Person1#가 ",
        r"\b과\s": "#Person1#과 ",
        r"\b와\s": "#Person1#와 ",
        r"\b에게\s": "#Person1#에게 ",
        r"\b한테\s": "#Person1#한테 ",
    }
    for pattern, replacement in tag_insert_rules.items():
        text = re.sub(pattern, replacement, text)

    # 6) 연속 중복 태그 제거 2차 (보완 후에도 중복 남을 수 있음)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(#Person\d#)(?:\s*\1)+", r"\1", text)

    # 7) 고도화된 중복 문장 제거
    def _advanced_dedup(s: str) -> str:
        sents = re.split(r"[.!?]", s)
        seen = set(); kept = []
        for sent in sents:
            sent_clean = sent.strip()
            if not sent_clean or len(sent_clean) < 3:  # 너무 짧은 문장 제거
                continue
                
            # 의미적 유사성 기반 중복 제거 (간단한 토큰 오버랩)
            is_duplicate = False
            sent_tokens = set(sent_clean.split())
            for prev_sent in kept:
                prev_tokens = set(prev_sent.split())
                if len(sent_tokens & prev_tokens) / len(sent_tokens | prev_tokens) > 0.7:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                kept.append(sent_clean)
        return ". ".join(kept)

    text = _advanced_dedup(text)

    # 8) 반복 구문 패턴 제거
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # 단어 반복 제거
    text = re.sub(r'(.{10,}?)\1+', r'\1', text)    # 긴 구문 반복 제거
    
    # 9) 공백 정리
    text = " ".join(text.split())

    # 10) 문장 품질 개선
    # 불완전한 문장 완성
    if text and not re.search(r'[.!?]$', text):
        if text.endswith(('다', '요', '음', '함', '됨', '임')):
            text += "."
        elif re.search(r'(받았|했|갔|왔|됐|했습니다)$', text):
            text += "."
        else:
            text += "습니다."
    
    # 11) 길이 최적화 (너무 짧거나 긴 요약 조정)
    if len(text) < 10:
        text = text + " 대화가 이루어졌습니다."
    elif len(text) > 200:
        # 첫 두 문장만 유지
        sentences = re.split(r'[.!?]', text)
        if len(sentences) > 2:
            text = '. '.join(sentences[:2]) + '.'

    return text.strip() 