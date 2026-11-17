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

    # True cross-script homoglyphs (Cyrillic, Greek, lookalikes) - NO Latin letters or pure digits here!
    HOMOGLYPH_MAP = {
        'а': 'a', 'а́': 'a', 'â': 'a', 'à': 'a', 'á': 'a', 'ä': 'a', 'α': 'a',
        'в': 'b', 'β': 'b',
        'с': 'c', 'ç': 'c', '¢': 'c', '©': 'c',
        'ԁ': 'd', 'đ': 'd',
        'е': 'e', 'е́': 'e', 'ê': 'e', 'è': 'e', 'é': 'e', 'ë': 'e', 'є': 'e', '€': 'e',
        'ƒ': 'f',
        'һ': 'h', 'н': 'h',
        'і': 'i', 'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
        'ј': 'j',
        'к': 'k',
        'м': 'm',
        'п': 'n', 'ñ': 'n',
        'о': 'o', 'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ø': 'o', 'θ': 'o',
        'р': 'p', 'ρ': 'p',
        'г': 'r', '®': 'r',
        'ѕ': 's', '§': 's',
        'т': 't', 'τ': 't',
        'υ': 'u', 'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'µ': 'u',
        'х': 'x', '×': 'x',
        'у': 'y', 'ý': 'y', 'ÿ': 'y', 'γ': 'y'
    }

    # Leetspeak mappings applied ONLY inside words containing alphabetic characters
    LEET_IN_WORD = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b',
        '@': 'a',
        '$': 's',
        '!': 'i'
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
        """Replaces cross-script homoglyphs with standard Latin characters."""
        if not text:
            return ""
        
        normalized = unicodedata.normalize('NFKD', text)
        result = []
        for char in normalized:
            lower_char = char.lower()
            if lower_char in cls.HOMOGLYPH_MAP:
                rep = cls.HOMOGLYPH_MAP[lower_char]
                result.append(rep.upper() if char.isupper() else rep)
            else:
                result.append(char)
        return "".join(result)

    @classmethod
    def decode_word_leetspeak(cls, text: str) -> str:
        """
        Decodes leetspeak digits & symbols ONLY when embedded inside words flanked by letters.
        Protects standalone numbers, percentages, dates, and currency (e.g. 2020, 100%, $37.50).
        """
        if not text:
            return ""

        def _decode_token(match):
            token = match.group(0)
            # If token is pure digits or pure punctuation/currency, leave it intact
            has_letters = bool(re.search(r'[a-zA-Z]', token))
            has_substitutions = any(c in cls.LEET_IN_WORD for c in token)
            
            if has_letters and has_substitutions:
                res = []
                for char in token:
                    if char in cls.LEET_IN_WORD:
                        res.append(cls.LEET_IN_WORD[char])
                    else:
                        res.append(char)
                return "".join(res)
            return token

        # Split on whitespace while preserving punctuation boundaries
        tokens = text.split(" ")
        cleaned_tokens = [_decode_token(re.match(r'\S+', tok)) if tok.strip() else tok for tok in tokens]
        return " ".join([t if isinstance(t, str) else t.group(0) for t in cleaned_tokens])

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
        step4 = cls.decode_word_leetspeak(step3)
        
        # Clean extra spaces
        return re.sub(r'\s+', ' ', step4).strip()


if __name__ == "__main__":
    test_cases = [
        "h@t3 speech in d1sg u1s3",
        "r.a.c.i.s.t meme",
        "f\u200bu\u200bc\u200bk off",
        "Cyrillic faked: һаte ѕреесh"
    ]
    for tc in test_cases:
        print(f"Original: {tc}  --->  Cleaned: {TextDeobfuscator.clean(tc)}")
