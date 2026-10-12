import unicodedata
import re

class TextDeobfuscator:
    """
    Normalizes text obfuscated using cloaking tricks:
    - Homoglyph replacement (Cyrillic/Greek lookalikes)
    - Leetspeak & symbol decoding (@ -> a, $ -> s, 1 -> i/l, 3 -> e, 0 -> o, 4 -> a, 5 -> s, 7 -> t, ! -> i)
    - Removal of zero-width and invisible control characters
    - De-spacing / de-punctualizing character-inserted words (e.g., h.a.t.e -> hate)
    """

    HOMOGLYPH_MAP = {
        'а': 'a', 'а́': 'a', 'â': 'a', 'à': 'a', 'á': 'a', 'ä': 'a', 'α': 'a', '@': 'a', '4': 'a',
        'в': 'b', 'β': 'b', '8': 'b',
        'с': 'c', 'ç': 'c', '¢': 'c', '©': 'c',
        'ԁ': 'd', 'đ': 'd',
        'е': 'e', 'е́': 'e', 'ê': 'e', 'è': 'e', 'é': 'e', 'ë': 'e', 'є': 'e', '3': 'e', '€': 'e',
        'f': 'f', 'ƒ': 'f',
        'g': 'g', '9': 'g',
        'һ': 'h', 'н': 'h', '#': 'h',
        'і': 'i', 'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', '1': 'i', '!': 'i', '|': 'i',
        'ј': 'j',
        'к': 'k',
        'l': 'l', '1': 'l', '|': 'l',
        'м': 'm',
        'п': 'n', 'ñ': 'n',
        'о': 'o', 'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', '0': 'o', 'ø': 'o', 'θ': 'o',
        'р': 'p', 'ρ': 'p',
        'q': 'q',
        'г': 'r', '®': 'r',
        'ѕ': 's', '$': 's', '5': 's', '§': 's',
        'т': 't', 'τ': 't', '7': 't', '+': 't',
        'υ': 'u', 'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'µ': 'u', 'v': 'u',
        'w': 'w', 'vv': 'w',
        'х': 'x', '×': 'x',
        'у': 'y', 'ý': 'y', 'ÿ': 'y', 'γ': 'y',
        'z': 'z', '2': 'z', '%': 'z'
    }

    ZERO_WIDTH_CHARS = re.compile(r'[\u200B-\u200D\uFEFF\u200E\u200F\u00AD]')
    REPEATED_PUNCTUATION_WORD = re.compile(r'\b(?:[a-zA-Z0-9][._\-\*~@\$#!+]){2,}[a-zA-Z0-9]\b')

    @classmethod
    def remove_zero_width(cls, text: str) -> str:
        """Strips invisible zero-width unicode characters."""
        if not text:
            return ""
        return cls.ZERO_WIDTH_CHARS.sub('', text)

    @classmethod
    def normalize_homoglyphs(cls, text: str) -> str:
        """Replaces homoglyphs and leetspeak symbols with standard Latin characters."""
        if not text:
            return ""
        
        # NFKD normalization first
        normalized = unicodedata.normalize('NFKD', text)
        result = []
        
        for char in normalized:
            lower_char = char.lower()
            if lower_char in cls.HOMOGLYPH_MAP:
                # Keep case matching if possible
                rep = cls.HOMOGLYPH_MAP[lower_char]
                result.append(rep.upper() if char.isupper() else rep)
            else:
                result.append(char)
                
        return "".join(result)

    @classmethod
    def collapse_inserted_delimiters(cls, text: str) -> str:
        """Collapses obfuscated spaced/punctuated words like 'h.a.t.e' or 'f_u_c_k'."""
        if not text:
            return ""

        def _clean_match(match):
            raw = match.group(0)
            return re.sub(r'[._\-\*~@\$#!+]', '', raw)

        return cls.REPEATED_PUNCTUATION_WORD.sub(_clean_match, text)

    @classmethod
    def clean(cls, text: str) -> str:
        """Full de-obfuscation pipeline."""
        if not text or not isinstance(text, str):
            return ""
        
        step1 = cls.remove_zero_width(text)
        step2 = cls.collapse_inserted_delimiters(step1)
        step3 = cls.normalize_homoglyphs(step2)
        
        # Clean extra spaces
        return re.sub(r'\s+', ' ', step3).strip()


if __name__ == "__main__":
    test_cases = [
        "h@t3 speech in d1sg u1s3",
        "r.a.c.i.s.t meme",
        "f\u200bu\u200bc\u200bk off",
        "Cyrillic faked: һаte ѕреесh"
    ]
    for tc in test_cases:
        print(f"Original: {tc}  --->  Cleaned: {TextDeobfuscator.clean(tc)}")
