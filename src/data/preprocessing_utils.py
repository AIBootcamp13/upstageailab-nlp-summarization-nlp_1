# preprocessing_utils.py는 전처리 함수들만 정의된 "라이브러리" 역할을 합니다.
# 데이터 전처리 과정은 다음과 같습니다.
# 1. 쌍따옴표 제거
# 2. 불필요한 토큰 제거
# 3. 개인정보 마스킹
# 4. 화자 정보 치환
# 5. 상투어 제거
# 6. 한 줄 평탄화

import re

# 제거할 토큰
REMOVE_TOKENS = ["<usr>", "<s>", "</s>", "<pad>"]

# 의미 없는 상투어 (중복 제거)
GENERIC_PHRASES = {"네", "알겠습니다", "음", "그렇군요", "아 네", "네네", "응"}

# 개인정보 패턴 -> 마스킹
PII_MASK = {
    r"\d{2,3}-\d{3,4}-\d{4}": "#PhoneNumber#",
    r"\d{6}-\d{7}": "#SSN#",
    r"\d{5,6}": "#AccountNumber#",
    r"[A-Z]{2}\d{7}": "#PassportNumber#",
    r"\w+@\w+\.\w+": "#Email#",
    r"\d{2,4} [가-힣]+로 [\d\-]+": "#Address#",
    r"[가-힣]{2,4}병원": "#Name#",
    r"[A-Z0-9]{7,}": "#LicenseNumber#",
}

def clean_text(text: str) -> str:
    """쌍따옴표 제거 + 토큰 제거 + 개인정보 마스킹"""
    if not isinstance(text, str):
        return ""
    
    # 0. 양쪽 쌍따옴표 제거
    text = text.strip('"')

    # 1. 불필요한 토큰 제거
    for token in REMOVE_TOKENS:
        text = text.replace(token, "")
    
    # 2. 개인정보 마스킹
    for pattern, repl in PII_MASK.items():
        text = re.sub(pattern, repl, text)

    return text.strip()


def flatten_dialogue(text: str) -> str:
    """#PersonN#: 형태 유지 + 상투어 제거 + 한 줄 평탄화 (정답 스타일과 일치)"""
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    flattened = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # #PersonN#: 형태 유지 (정답과 일치시키기 위해)
        # 단순히 공백과 콜론 정리만 수행
        line = re.sub(r"#Person(\d+)#\s*:\s*", r"#Person\1#: ", line)

        if line in GENERIC_PHRASES:
            continue

        flattened.append(line)

    return " ".join(flattened) 