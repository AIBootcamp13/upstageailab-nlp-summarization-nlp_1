import re
import pandas as pd

# ✅ 1. 특수문자 제거 리스트
REMOVE_TOKENS = ['@', '%', '$', '&', '+', '`', '·', '*', '<', '>', '=', '^', '_', '\\', '[', ']', '{', '}']
# ✅ 2. 의미 치환 사전 (감정표현 → 단어)
EMOTION_MAP = {
    'ㅋㅋ': '웃음',
    'ㅎㅎ': '웃음',
    'ㅠㅠ': '슬픔',
    'ㅜㅜ': '슬픔',
    'ㅇㅇ': '응',
    'ㄱㄱ': '가자',
    'ㄴㄴ': '아니',
    'ㅠ': '슬픔',
    'ㅜ': '슬픔'
}
# ✅ 3. 정규표현식
MASK_TOKEN_PATTERN = re.compile(r"#\w+#")
PERSON_TOKEN_PATTERN = re.compile(r"#Person\d+#")

# ✅ 4. 전처리 함수
def clean_text(text: str) -> str:
    # 1) 감정 표현 치환
    for token, replacement in EMOTION_MAP.items():
        text = text.replace(token, replacement)

    # 2) 특수문자 제거
    for token in REMOVE_TOKENS:
        text = text.replace(token, '')

    # 3) 중복된 감정 표현 정규화
    text = re.sub(r"(웃음){2,}", "웃음", text)
    text = re.sub(r"(슬픔){2,}", "슬픔", text)

    return text.strip()

# ✅ 5. 요약 비율 필터링 (dialogue 대비 summary 길이 비율)
def filter_by_length(df, min_ratio=0.12, max_ratio=0.4):
    dialogue_len = df['dialogue'].apply(len)
    summary_len = df['summary'].apply(len)
    ratio = summary_len / dialogue_len
    return df[(ratio >= min_ratio) & (ratio <= max_ratio)]

# ✅ 6. 마스킹 토큰 추출 (선택: 분석용)
def extract_masked_tokens(text):
    return MASK_TOKEN_PATTERN.findall(text)

def extract_person_tokens(text):
    return PERSON_TOKEN_PATTERN.findall(text)

# ✅ 7. 전체 전처리 적용 함수
def apply_preprocessing(df: pd.DataFrame, is_train=True):
    df['dialogue'] = df['dialogue'].apply(clean_text)
    if is_train and 'summary' in df.columns:
        df['summary'] = df['summary'].apply(clean_text)
        df = filter_by_length(df)
    return df

# ✅ 8. 사용 예시
if __name__ == "__main__":
    # CSV 경로 설정
    train_path = "./data/train.csv"
    dev_path = "./data/dev.csv"
    test_path = "./data/test.csv"

    # 불러오기
    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)

    # 전처리 적용
    train_clean = apply_preprocessing(train, is_train=True)
    dev_clean = apply_preprocessing(dev, is_train=True)
    test_clean = apply_preprocessing(test, is_train=False)

    # 저장
    train_clean.to_csv("./data/train_clean.csv", index=False)
    dev_clean.to_csv("./data/dev_clean.csv", index=False)
    test_clean.to_csv("./data/test_clean.csv", index=False)

    print("전처리 완료 ✅")
