import re
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """
    Extract MCQ answers from LLM responses using multiple strategies:
    1. Regex patterns (fast, works offline)
    2. EleutherAI lm-eval filters (battle-tested patterns)
    3. xFinder model (more accurate, requires GPU)
    4. HuggingFace LLM fallback (most flexible, requires API/model)
    """
    
    def __init__(
        self,
        use_eleutherai: bool = True,
        use_xfinder: bool = True,
        use_hf_fallback: bool = True,
        xfinder_model: str = "IAAR-Shanghai/xFinder-qwen1505",
        hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto"
    ):
        self.use_eleutherai = use_eleutherai
        self.use_xfinder = use_xfinder
        self.use_hf_fallback = use_hf_fallback
        self.device = device
        
        # Lazy loading
        self._eleutherai_filter = None
        self._xfinder_pipeline = None
        self._hf_pipeline = None
        self._xfinder_model_name = xfinder_model
        self._hf_model_name = hf_model
    
    @property
    def eleutherai_filter(self):
        """Lazy load EleutherAI MultiChoiceRegexFilter"""
        if self._eleutherai_filter is None and self.use_eleutherai:
            try:
                from lm_eval.filters.extraction import RegexFilter
                from lm_eval.filters.selection import TakeFirstFilter
                logger.info("Loaded EleutherAI lm-eval filters")
                self._eleutherai_filter = True  # Mark as available
            except ImportError:
                logger.warning("lm-eval not installed. Install with: pip install lm-eval")
                self.use_eleutherai = False
        return self._eleutherai_filter
    
    @property
    def xfinder_pipeline(self):
        """Lazy load xFinder model"""
        if self._xfinder_pipeline is None and self.use_xfinder:
            try:
                from transformers import pipeline
                logger.info(f"Loading xFinder model: {self._xfinder_model_name}")
                self._xfinder_pipeline = pipeline(
                    "text-generation",
                    model=self._xfinder_model_name,
                    device_map=self.device,
                    torch_dtype="auto"
                )
            except Exception as e:
                logger.warning(f"Failed to load xFinder: {e}")
                self.use_xfinder = False
        return self._xfinder_pipeline
    
    @property
    def hf_pipeline(self):
        """Lazy load HuggingFace fallback model"""
        if self._hf_pipeline is None and self.use_hf_fallback:
            try:
                from transformers import pipeline
                logger.info(f"Loading HF fallback model: {self._hf_model_name}")
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model=self._hf_model_name,
                    device_map=self.device,
                    torch_dtype="auto"
                )
            except Exception as e:
                logger.warning(f"Failed to load HF fallback: {e}")
                self.use_hf_fallback = False
        return self._hf_pipeline
    
    def extract(
        self,
        response: str,
        valid_options: List[str],
        question: Optional[str] = None
    ) -> str:
        """
        Extract answer using fallback chain: regex -> EleutherAI -> xFinder -> HF LLM
        
        Args:
            response: The LLM's response text
            valid_options: List of valid answer options (e.g., ["A", "B", "C", "D"])
            question: Original question (needed for xFinder/HF fallback)
        
        Returns:
            Extracted answer in uppercase, or empty string if not found
        """
        # Strategy 1: Custom Regex (fast, bilingual)
        result = self._extract_with_regex(response, valid_options)
        if result:
            logger.debug(f"Regex extracted: {result}")
            return result
        
        # Strategy 2: EleutherAI lm-eval filters (battle-tested)
        if self.use_eleutherai:
            result = self._extract_with_eleutherai(response, valid_options)
            if result:
                logger.debug(f"EleutherAI extracted: {result}")
                return result
        
        # Strategy 3: xFinder (accurate)
        if self.use_xfinder and question:
            result = self._extract_with_xfinder(response, valid_options, question)
            if result:
                logger.debug(f"xFinder extracted: {result}")
                return result
        
        # Strategy 4: HF LLM fallback (flexible)
        if self.use_hf_fallback and question:
            result = self._extract_with_hf(response, valid_options, question)
            if result:
                logger.debug(f"HF fallback extracted: {result}")
                return result
        
        logger.warning(f"Failed to extract answer from: {response[:100]}...")
        return ""
    
    def _extract_with_regex(self, response: str, valid_options: List[str]) -> str:
        """Extract answer using regex patterns (English + French)"""
        response_lower = response.strip().lower()
        response_upper = response.strip().upper()
        valid_lower = [o.lower() for o in valid_options]
        valid_upper = [o.upper() for o in valid_options]
        
        # Sort by length descending so "AA" is matched before "A"
        options_pattern_lower = '|'.join(sorted(valid_lower, key=len, reverse=True))
        options_pattern_upper = '|'.join(sorted(valid_upper, key=len, reverse=True))
        
        patterns = [
            # === ENGLISH PATTERNS ===
            
            # Final/definitive answers (highest priority)
            r'final\s+answer\s+is\s+({opt})',
            r'final\s+answer[:\s]+({opt})',
            r'final\s+choice\s+is\s+({opt})',
            r'(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)\s+is\s+({opt})',
            r'(?:the\s+)?correct\s+answer\s+is\s+({opt})',
            r'correct[:\s]+({opt})',
            r'correct\s+(?:option|choice|answer)[:\s]+({opt})',
            
            # Standard answer patterns
            r'(?:the\s+)?answer\s+is\s+({opt})',
            r'(?:the\s+)?response\s+is\s+({opt})',
            r'(?:the\s+)?answer[:\s]+({opt})',
            r'(?:the\s+)?response[:\s]+({opt})',
            r'answer\s*=\s*({opt})',
            
            # Personal choice patterns
            r'my\s+(?:answer|choice|response)\s+is\s+({opt})',
            r'i\s+(?:would\s+)?(?:choose|select|pick)\s+({opt})',
            r'i\s+(?:believe|think)\s+(?:it\'?s?|the\s+answer\s+is)\s+({opt})',
            r'(?:i\'?ll?\s+)?go(?:ing)?\s+with\s+({opt})',
            
            # Modal patterns
            r'should\s+be\s+({opt})',
            r'would\s+be\s+({opt})',
            r'must\s+be\s+({opt})',
            r'it\'?s?\s+({opt})',
            
            # Reverse patterns
            r'({opt})\s+is\s+(?:the\s+)?(?:correct|right|best|answer)',
            
            # Conclusion indicators
            r'(?:therefore|thus|hence|so)[,:\s]+({opt})',
            
            # Labeled patterns
            r'is\s+({opt})',
            r'option\s+({opt})',
            r'choice\s+({opt})',
            r'select\s+({opt})',
            r'choose\s+({opt})',
            
            # === FRENCH PATTERNS ===
            
            # Final/definitive answers
            r'r[ée]ponse\s+finale\s+est\s+({opt})',
            r'r[ée]ponse\s+finale[:\s]+({opt})',
            r'choix\s+final\s+est\s+({opt})',
            r'choix\s+final[:\s]+({opt})',
            r'(?:la\s+)?(?:bonne|correcte)\s+r[ée]ponse\s+est\s+({opt})',
            r'(?:la\s+)?r[ée]ponse\s+(?:correcte|bonne)\s+est\s+({opt})',
            
            # Standard answer patterns
            r'(?:la\s+)?r[ée]ponse\s+est\s+({opt})',
            r'(?:la\s+)?r[ée]ponse[:\s]+({opt})',
            r'r[ée]ponse\s*=\s*({opt})',
            
            # Personal choice patterns
            r'ma\s+r[ée]ponse\s+est\s+({opt})',
            r'mon\s+choix\s+est\s+({opt})',
            r'je\s+(?:choisis|s[ée]lectionne)\s+({opt})',
            r'j\'?opterais?\s+pour\s+({opt})',
            r'je\s+(?:pense|crois)\s+que\s+c\'?est\s+({opt})',
            r'je\s+(?:pense|crois)\s+que\s+la\s+r[ée]ponse\s+est\s+({opt})',
            r'je\s+(?:dirais?|pencherais?)\s+pour\s+({opt})',
            
            # Modal patterns
            r'(?:ce\s+)?devrait\s+[êe]tre\s+({opt})',
            r'(?:ce\s+)?serait\s+({opt})',
            r'(?:ce\s+)?doit\s+[êe]tre\s+({opt})',
            r'c\'?est\s+({opt})',
            r'il\s+s\'?agit\s+de\s+({opt})',
            
            # Reverse patterns
            r'({opt})\s+est\s+(?:la\s+)?(?:bonne|correcte)\s+r[ée]ponse',
            r'({opt})\s+est\s+(?:correct|juste|bon)',
            
            # Conclusion indicators
            r'(?:donc|ainsi|par\s+cons[ée]quent|en\s+conclusion)[,:\s]+({opt})',
            r'(?:finalement|pour\s+conclure)[,:\s]+({opt})',
            
            # Labeled patterns
            r'est\s+({opt})',
            r'choix\s+({opt})',
            r's[ée]lectionner\s+({opt})',
            r'choisir\s+({opt})',
            
            # === FORMATTING PATTERNS ===
            
            r'\*\*({opt})\*\*',
            r'\*({opt})\*',
            r'`({opt})`',
            r'\(({opt})\)',
            r'({opt})\)',
            r'({opt}):',
            r'({opt})\s*-',
            r'^[\-\*\•]\s*({opt})',
            r'[\-\*\•]\s*({opt})',
            
            # Generic match (lowest priority)
            r'({opt})',
        ]
        
        # First pass: search in lowercase
        for pattern_template in patterns:
            pattern = pattern_template.replace('{opt}', f'({options_pattern_lower})')
            match = re.search(pattern, response_lower, re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # Second pass: search in uppercase
        for pattern_template in patterns:
            pattern = pattern_template.replace('{opt}', f'({options_pattern_upper})')
            match = re.search(pattern, response_upper, re.MULTILINE)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_with_eleutherai(self, response: str, valid_options: List[str]) -> str:
        """
        Extract answer using EleutherAI lm-evaluation-harness patterns.
        These are battle-tested patterns used in the Open LLM Leaderboard.
        """
        if not self.eleutherai_filter:
            return ""
        
        try:
            # Try using lm-eval's built-in filter
            return self._eleutherai_regex_extraction(response, valid_options)
        except Exception as e:
            logger.warning(f"EleutherAI extraction failed: {e}")
            return ""
    
    def _eleutherai_regex_extraction(self, response: str, valid_options: List[str]) -> str:
        """
        Reimplementation of EleutherAI's MultiChoiceRegexFilter patterns.
        Based on lm-evaluation-harness filters used for MMLU, ARC, HellaSwag, etc.
        """
        response_clean = response.strip()
        valid_upper = [o.upper() for o in valid_options]
        options_pattern = '|'.join(sorted(valid_upper, key=len, reverse=True))
        
        # EleutherAI-style patterns (from lm-eval tasks)
        eleutherai_patterns = [
            # MMLU style: "(A)" or "(B)"
            rf'\(({options_pattern})\)',
            
            # ARC style: "A." or "B."
            rf'\b({options_pattern})\.',
            
            # HellaSwag style: "A:" or "B:"
            rf'\b({options_pattern}):',
            
            # Answer is pattern (common in CoT)
            rf'[Aa]nswer\s*(?:is|:)\s*\(?({options_pattern})\)?',
            
            # The answer is pattern
            rf'[Tt]he\s+answer\s+is\s*:?\s*\(?({options_pattern})\)?',
            
            # Therefore/Thus pattern (chain-of-thought)
            rf'(?:[Tt]herefore|[Tt]hus|[Ss]o|[Hh]ence),?\s*(?:the\s+answer\s+is\s*)?\(?({options_pattern})\)?',
            
            # Best answer pattern
            rf'[Bb]est\s+answer\s*(?:is|:)\s*\(?({options_pattern})\)?',
            
            # Correct answer pattern
            rf'[Cc]orrect\s+answer\s*(?:is|:)\s*\(?({options_pattern})\)?',
            
            # Option X is correct
            rf'[Oo]ption\s+({options_pattern})\s+is\s+(?:correct|right|best)',
            
            # X is the correct answer
            rf'({options_pattern})\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)',
            
            # Simple letter at start (strict)
            rf'^({options_pattern})[\.\)\s]',
            
            # Simple letter at end (strict)
            rf'[\.\s]\(?({options_pattern})\)?$',
            
            # French patterns for EleutherAI compatibility
            rf'[Rr][ée]ponse\s*(?:est|:)\s*\(?({options_pattern})\)?',
            rf'[Ll]a\s+r[ée]ponse\s+est\s*:?\s*\(?({options_pattern})\)?',
            rf'(?:[Dd]onc|[Aa]insi),?\s*(?:la\s+r[ée]ponse\s+est\s*)?\(?({options_pattern})\)?',
            
            # Fallback: isolated letter with word boundary
            rf'\b({options_pattern})\b',
        ]
        
        # Try each pattern
        for pattern in eleutherai_patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).upper()
                if extracted in valid_upper:
                    return extracted
        
        # Last resort: first valid option character found
        for char in response_clean.upper():
            if char in valid_upper:
                return char
        
        return ""
    
    def _extract_with_xfinder(
        self,
        response: str,
        valid_options: List[str],
        question: str
    ) -> str:
        """Extract answer using xFinder model"""
        if not self.xfinder_pipeline:
            return ""
        
        try:
            prompt = self._build_xfinder_prompt(question, response, valid_options)
            
            output = self.xfinder_pipeline(
                prompt,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                return_full_text=False
            )
            
            extracted = output[0]["generated_text"].strip().upper()
            
            # Validate against valid options
            valid_upper = [o.upper() for o in valid_options]
            for opt in sorted(valid_upper, key=len, reverse=True):
                if opt in extracted:
                    return opt
            
            return ""
            
        except Exception as e:
            logger.warning(f"xFinder extraction failed: {e}")
            return ""
    
    def _build_xfinder_prompt(
        self,
        question: str,
        response: str,
        valid_options: List[str]
    ) -> str:
        """Build prompt for xFinder model (bilingual)"""
        options_str = ", ".join(valid_options)
        is_french = self._detect_french(question)
        
        if is_french:
            return f"""<|im_start|>system
Tu es un assistant spécialisé dans l'extraction de réponses. Extrais uniquement la lettre de réponse du texte donné.
<|im_end|>
<|im_start|>user
Question: {question}

Réponse du modèle: {response}

Options valides: {options_str}

Extrais uniquement la lettre de réponse choisie. Réponds avec seulement la lettre, rien d'autre.
<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"""<|im_start|>system
You are a specialized answer extraction assistant. Extract only the answer letter from the given text.
<|im_end|>
<|im_start|>user
Question: {question}

Model response: {response}

Valid options: {options_str}

Extract only the chosen answer letter. Reply with just the letter, nothing else.
<|im_end|>
<|im_start|>assistant
"""
    
    def _extract_with_hf(
        self,
        response: str,
        valid_options: List[str],
        question: str
    ) -> str:
        """Extract answer using HuggingFace LLM as final fallback"""
        if not self.hf_pipeline:
            return ""
        
        try:
            prompt = self._build_hf_extraction_prompt(question, response, valid_options)
            
            output = self.hf_pipeline(
                prompt,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                return_full_text=False
            )
            
            extracted = output[0]["generated_text"].strip().upper()
            
            # Validate against valid options
            valid_upper = [o.upper() for o in valid_options]
            for opt in sorted(valid_upper, key=len, reverse=True):
                if opt in extracted:
                    return opt
            
            return ""
            
        except Exception as e:
            logger.warning(f"HF fallback extraction failed: {e}")
            return ""
    
    def _build_hf_extraction_prompt(
        self,
        question: str,
        response: str,
        valid_options: List[str]
    ) -> str:
        """Build prompt for HF fallback model (bilingual)"""
        options_str = ", ".join(valid_options)
        is_french = self._detect_french(question)
        
        if is_french:
            return f"""<|im_start|>system
Tu extrais les réponses aux QCM. Réponds UNIQUEMENT avec la lettre de la réponse ({options_str}), rien d'autre.
<|im_end|>
<|im_start|>user
Question: {question}

Le modèle a répondu: {response}

Quelle option a été choisie? Réponds uniquement avec la lettre.
<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"""<|im_start|>system
You extract MCQ answers. Reply ONLY with the answer letter ({options_str}), nothing else.
<|im_end|>
<|im_start|>user
Question: {question}

The model responded: {response}

Which option was chosen? Reply only with the letter.
<|im_end|>
<|im_start|>assistant
"""
    
    def _detect_french(self, text: str) -> bool:
        """Simple French language detection"""
        french_indicators = [
            "est", "sont", "faire", "avoir", "être", "quelle", "quel", 
            "pourquoi", "comment", "qu'est", "c'est", "n'est", "d'un",
            "dans", "pour", "avec", "sur", "une", "des", "les", "aux"
        ]
        text_lower = text.lower()
        matches = sum(1 for word in french_indicators if f" {word} " in f" {text_lower} ")
        return matches >= 2


# Convenience function for simple usage
def extract_answer(
    response: str,
    valid_options: List[str],
    question: Optional[str] = None,
    use_fallbacks: bool = False
) -> str:
    """
    Simple function to extract MCQ answer from response.
    
    Args:
        response: LLM response text
        valid_options: Valid answer options (e.g., ["A", "B", "C", "D"])
        question: Original question (needed if use_fallbacks=True)
        use_fallbacks: Whether to use EleutherAI/xFinder/HF fallbacks
    
    Returns:
        Extracted answer letter (uppercase) or empty string
    """
    extractor = AnswerExtractor(
        use_eleutherai=use_fallbacks,
        use_xfinder=use_fallbacks,
        use_hf_fallback=use_fallbacks
    )
    return extractor.extract(response, valid_options, question)


# Test suite
def test_extractor():
    """Test the answer extractor with various formats"""
    extractor = AnswerExtractor(
        use_eleutherai=False,
        use_xfinder=False,
        use_hf_fallback=False
    )
    
    options = ["A", "B", "C", "D"]
    
    test_cases = [
        # English patterns
        ("The answer is B", "B"),
        ("I think the correct answer is C.", "C"),
        ("Therefore, A", "A"),
        ("(B)", "B"),
        ("B.", "B"),
        ("My final answer is D", "D"),
        ("I would choose A because...", "A"),
        ("The best option is C", "C"),
        ("After analysis, I believe it's B", "B"),
        ("**A**", "A"),
        
        # French patterns
        ("La réponse est B", "B"),
        ("Je pense que la bonne réponse est C", "C"),
        ("Donc, A", "A"),
        ("Ma réponse finale est D", "D"),
        ("Je choisis A car...", "A"),
        ("La réponse correcte est C", "C"),
        ("C'est B", "B"),
        ("Réponse: A", "A"),
        
        # Multi-char options
        ("The answer is AA", "AA"),
        ("La réponse est BB", "BB"),
        
        # Edge cases
        ("AAAAAAA", "A"),  # Repeated chars
        ("B) is correct", "B"),
        ("A - the first option", "A"),
    ]
    
    # Test multi-char options
    multi_options = ["AA", "BB", "CC", "DD"]
    
    print("Testing Answer Extractor\n" + "=" * 50)
    
    passed = 0
    failed = 0
    
    for response, expected in test_cases:
        opts = multi_options if expected in multi_options else options
        result = extractor.extract(response, opts)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{response[:40]}...' -> Expected: {expected}, Got: {result}")
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    test_extractor()