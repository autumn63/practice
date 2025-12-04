import re
from .utils import normalize_text
from data.badwords import BASE_BADWORDS, NUMBER_LIKE_PATTERNS

class ProfanityFilter:
    def __init__(self, mask_token="***"):
        self.mask_token = mask_token
        
        # 욕설 패턴 생성 (노이즈 허용)
        self.patterns = [self._build_fuzzy_pattern(w) for w in BASE_BADWORDS]
        
        # 숫자/기호 기반 욕설 패턴 추가
        self.patterns.extend([
            re.compile(p, re.IGNORECASE) for p in NUMBER_LIKE_PATTERNS
        ])

    def clean(self, text: str) -> str:
        norm = normalize_text(text)
        cleaned = norm
        for p in self.patterns:
            cleaned = p.sub(self.mask_token, cleaned)
        return cleaned

    def has_profanity(self, text: str) -> bool:
        norm = normalize_text(text)
        return any(p.search(norm) for p in self.patterns)

    def _build_fuzzy_pattern(self, badword: str) -> re.Pattern:
        escaped = [re.escape(ch) for ch in badword]
        noise = r"(?:\s|[\W_]){0,3}"
        fuzzy = noise.join(escaped)
        return re.compile(fuzzy, re.IGNORECASE)