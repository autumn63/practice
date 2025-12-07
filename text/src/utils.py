import re
import unicodedata

def _normalize(self, text: str) -> str:
        
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