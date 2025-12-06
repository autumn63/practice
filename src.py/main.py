import re
import unicodedata
import os

class ProfanityFilter:
    def __init__(self, mask_token="***"):
        self.mask_token = mask_token

        # 1) 기본 욕설/비속어 리스트 (예시 - 직접 확장해서 쓰셔야 합니다)
        #   너무 노골적인 건 줄였고, 실제에선 훨씬 더 많은 패턴이 필요합니다.
        base_badwords = [
            "씨발", "시발", "씹팔", "좆같", "존나", "존내", "병신", "병신같",
            "개새끼", "개색기", "개같", "쓰레기", "닥쳐",
            "ㅅㅂ", "ㅆㅂ", "ㅄ",'ㅈㄴ','^^ㅣ발',
        ]

        # 2) 각 단어를 '노이즈 허용 정규식'으로 변환
        #   예: 씨발 -> ㅆ*ㅣ*발 / ㅆ+ㅣ+발+ 이런 식으로 변형 허용
        self.patterns = [self._build_fuzzy_pattern(w) for w in base_badwords]

        # 3) 숫자/영문으로 대체되는 욕 표현도 일부 패턴으로 포함 (예시)
        #   18 -> ㅆㅂ / 십팔 욕으로 쓰이는 경우
        #   10발 -> 시발 변형으로 쓰이는 경우 등
        number_like_patterns = [
            r"1\s*8",         # 18
            r"10\s*발",       # 10발, 10발아...
        ]
        self.patterns.extend([re.compile(p, re.IGNORECASE) for p in number_like_patterns])

    # ------------------- 공개 API -------------------

    def clean(self, text: str) -> str:
        """
        욕설을 탐지하여 mask_token으로 치환한 텍스트를 반환
        """
        norm = self._normalize(text)
        cleaned = norm

        for pat in self.patterns:
            cleaned = pat.sub(self.mask_token, cleaned)

        return cleaned

    def has_profanity(self, text: str) -> bool:
        """
        욕설이 하나라도 포함되어 있으면 True
        """
        norm = self._normalize(text)
        return any(p.search(norm) for p in self.patterns)

    # ------------------- 내부 유틸 -------------------

    def _normalize(self, text: str) -> str:
        """
        - 유니코드 정규화
        - 전각/반각 통일
        - 영어 소문자화
        - 한글 자모가 띄어져 있는 경우 약간 정리 (아주 완벽하진 않음)
        - 불필요한 반복 문자 줄이기 (ㅋㅋㅋㅋ -> ㅋㅋ 정도)
        """
        # 유니코드 정규화
        text = unicodedata.normalize("NFC", text)

        # 영어 소문자
        text = text.lower()

        # 반복 문자 줄이기 (3번 이상 반복 -> 2번으로)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 흔한 웃음/방언/이모티콘은 관심 없으면 미리 정리 가능 (선택)
        # text = re.sub(r"[ㅋㅎ]+", "ㅋ", text)

        # 양쪽 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _build_fuzzy_pattern(self, badword: str) -> re.Pattern:
        """
        기본 욕 단어를 받아서, 중간에 노이즈(공백, 특수문자 등)가 끼어도
        인식할 수 있게 정규식 패턴으로 바꿉니다.
        예: '씨발' -> 'ㅆ\s*[\W_]*ㅣ\s*[\W_]*발' 느낌
        """
        # 자주 쓰이는 한글 욕의 경우 초성/중성 분리까지 할 수 있지만
        # 여기선 간단히: 글자 사이에 특수문자/공백 몇 개 허용 정도로 처리
        escaped = [re.escape(ch) for ch in badword]

        # 글자 하나마다 "중간에 잡음 허용" 패턴 끼워 넣기
        # \s* : 공백
        # [\W_]* : 영문/숫자 제외 특수문자들
        noise = r"(?:\s|[\W_]){0,3}"
        fuzzy = noise.join(escaped)

        pattern = re.compile(fuzzy, re.IGNORECASE)
        return pattern

# ----------------- 실행 및 파일 저장 로직 -----------------

def save_filtered_results(filter_instance, text_list, output_folder="filtered_results", filename="cleaned_log.txt"):
    """
    텍스트 리스트를 필터링하여 지정된 폴더의 파일로 저장하는 함수
    (줄 단위로 쪼개지 않고 텍스트 전체를 하나로 처리)
    """
    
    # 1. 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"폴더 생성 완료: {output_folder}")
    
    # 2. 파일 전체 경로 설정
    file_path = os.path.join(output_folder, filename)

    # 3. 파일 쓰기
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== 욕설 필터링 결과 로그 (통문장 처리) ===\n\n")
        
        for idx, text in enumerate(text_list, 1):
            text = text.strip()
            if not text: continue

            f.write(f"[Example {idx}]\n")
            
            # 전체 텍스트를 한 번에 처리
            # 주의: 현재 _normalize 로직에 의해 줄바꿈이 공백으로 바뀔 수 있습니다.
            cleaned_text = filter_instance.clean(text)
            is_bad = filter_instance.has_profanity(text)
            
            # 파일에 기록할 포맷
            f.write(f"원문:\n{text}\n\n")
            f.write(f"정제 결과:\n{cleaned_text}\n\n")
            f.write(f"욕 포함?: {'True' if is_bad else '정상'}\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"저장 완료 파일경로: {file_path}")



# ----------------- 사용 예시 -----------------

if __name__ == "__main__":
    f = ProfanityFilter()

    examples = [
        '''
헛개수?
헛수고
ㅈㄴ 잘하는데
와 조금 아쉽긴하네 진짜 못하긴한다.
나는 수박할게
수박은 진짜 아쉽긴하네 진짜 못하긴한다. 개병신이네
박자감!
와~~! ^^ㅣ발
박수
아까 잘한다며 창의력 넘친다하더니만
이런 씨발
이런 병신같은
'''
    ]



    for s in examples:
        print("원문:", s)
        print("정제:", f.clean(s))
        print("욕 포함?:", f.has_profanity(s))
        print("-" * 30)

    # 3. 바탕화면 경로 찾기
    desktop_path = os.path.join(os.path.expanduser("~"), "OneDrive", "바탕 화면", "filtered_results")

    # 4. 파일 저장 함수 실행 (output_folder를 바탕화면으로 지정)
    save_filtered_results(f, examples, output_folder=desktop_path)