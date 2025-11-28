import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ==========================================
# 1. 파일 로더 & 데이터 처리
# ==========================================
def load_fword_list(file_path):
    """
    업로드된 텍스트 파일을 읽어서 리스트로 변환합니다.
    """
    try:
        with open("C:/Users/minhs/OneDrive/바탕 화면/욕설필터링/fword_list.txt", 'r', encoding='utf-8') as f:
            # 줄바꿈 제거하고 리스트로 변환
            lines = f.readlines()
            # 공백 제거 및 빈 줄 제외
            fwords = [line.strip() for line in lines if line.strip()]
        print(f">> 욕설 리스트 파일 로드 완료: {len(fwords)}개 단어")
        return fwords
    except FileNotFoundError:
        print("!! 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return []

def create_training_data(fwords):
    """
    1. 온라인 실전 데이터 (UnSmile 데이터셋)
    2. 사용자 업로드 욕설 리스트 (fword_list.txt)
    이 두 가지를 합쳐서 강력한 학습 데이터를 만듭니다.
    """
    print(">> 데이터셋 병합 중...")
    
    # 1. 기본 실전 데이터 (UnSmile)
    url = "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_train_v1.0.tsv"
    base_df = pd.read_csv(url, delimiter='\t')
    base_df['label'] = base_df['clean'].apply(lambda x: 0 if x == 1 else 1)
    base_df = base_df[['문장', 'label']]
    base_df.columns = ['text', 'label']

    # 2. 업로드한 욕설 리스트를 학습 데이터로 증강 (Data Augmentation)
    # 리스트에 있는 단어들을 '욕설(1)' 라벨로 데이터프레임에 추가
    fword_data = pd.DataFrame({
        'text': fwords,
        'label': [1] * len(fwords) # 전량 욕설(1)로 라벨링
    })

    # 두 데이터 합치기
    final_df = pd.concat([base_df, fword_data], ignore_index=True)
    
    # 데이터 섞기 (Shuffle)
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    print(f">> 최종 학습 데이터 생성 완료: 총 {len(final_df)}개")
    return final_df

# ==========================================
# 2. AI 모델 (이전과 동일하지만 더 강력해짐)
# ==========================================
class ProfanityClassifier:
    def __init__(self):
        self.model = Pipeline([
            # ngram_range=(1, 3): 단어 3개 조합까지 봅니다 (더 정교함)
            ('vect', CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3))), 
            ('clf', MultinomialNB())
        ])
        self.is_trained = False

    def train(self, df):
        print(">> AI 모델 학습 시작...")
        self.model.fit(df['text'], df['label'])
        self.is_trained = True
        print(">> AI 모델 학습 완료!")

    def predict_prob(self, text):
        return self.model.predict_proba([text])[0][1]

# ==========================================
# 3. 스마트 순화기 (파일 리스트 연동)
# ==========================================
class TextPurifier:
    def __init__(self, fwords):
        self.fwords = fwords # 업로드된 욕설 리스트
        
        # 1. 고정 매핑 (자주 쓰는 건 예쁘게)
        self.mapping = {
            '놈': '친구', '년': '분', '새끼': '강아지',
            '미친': '상상력이 풍부한', 'ㅁㅊ': '깜짝 놀랄',
            '지랄': '재롱', '병신': '아픈 친구',
            '씨발': '이런', '시발': '이런', 'ㅅㅂ': '이런',
            '존나': '정말', '개': '많이', '죽어': '살아',
            '꺼져': '잠시만 안녕', '닥쳐': '쉿',
            '쓰레기': '재활용품'
        }
        
        # 2. 매핑에 없는 욕설이 들어오면 사용할 '랜덤 순화어'
        self.nice_words = ["(예쁜말)", "사랑둥이", "귀염둥이", "멋쟁이", "소중한 사람", "행복하세요"]

    def purify(self, text):
        purified = text
        
        # [1단계] 업로드된 리스트에 있는 단어 강제 치환 (Rule-based)
        # 긴 단어부터 치환해야 함 (예: '개새끼'를 먼저 잡고 '개'를 잡아야 함)
        sorted_fwords = sorted(self.fwords, key=len, reverse=True)
        
        for word in sorted_fwords:
            if word in purified:
                # 매핑된 게 있으면 그걸로, 없으면 랜덤 좋은 말로
                replacement = self.mapping.get(word, random.choice(self.nice_words))
                purified = purified.replace(word, replacement)
        
        return purified

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    # 1. 파일 로드 (같은 폴더에 fword_list.txt가 있어야 합니다)
    # 코랩이나 주피터 노트북이라면 경로를 맞춰주세요.
    fwords = load_fword_list('fword_list.txt')
    
    if not fwords:
        print("오류: fword_list.txt 파일을 찾을 수 없거나 비어있습니다.")
        # 테스트를 위한 더미 데이터
        fwords = ['나쁜말', '욕설'] 

    # 2. 데이터셋 생성 (실전 데이터 + 내 파일 데이터)
    df = create_training_data(fwords)
    
    # 3. 모델 학습
    classifier = ProfanityClassifier()
    classifier.train(df)
    
    # 4. 순화기 초기화
    purifier = TextPurifier(fwords)
    
    print("\n" + "="*50)
    print("🚀 커스텀 욕설 필터링 시스템 가동")
    print("파일에 있는 단어는 즉시 순화되고, 애매한 말은 AI가 판단합니다.")
    print("="*50)

    while True:
        user_input = input("\n[입력]: ")
        if user_input.lower() == 'q': break
        
        # 1. AI 확률 계산
        prob = classifier.predict_prob(user_input)
        
        # 2. 리스트에 있는 단어가 포함되어 있는지 직접 확인
        contains_bad_word = any(word in user_input for word in fwords)
        
        print(f" > 욕설 확률: {prob*100:.1f}%")

        # 3. 판단 로직: (확률이 높거나) OR (리스트에 있는 단어가 있거나)
        if prob >= 0.6 or contains_bad_word:
            print(" > 판정: 🚨 순화 대상")
            
            # 순화 실행
            clean_text = purifier.purify(user_input)
            print(f" > 순화: ✨ {clean_text}")
        else:
            print(" > 판정: ✅ 정상")