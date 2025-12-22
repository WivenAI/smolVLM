"""
QCM Evaluator - Evaluates ERP multiple choice questions
"""

from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
import re
from PIL import Image

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class QCMEvaluator(BaseEvaluator):
    """Evaluator for QCM (Multiple Choice Questions) on ERP screenshots"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_path = None
        self.image_dir = None

    def evaluate(self, model_path: str = None, max_samples: int = None,
                 dataset_path: str = None, image_dir: str = None) -> Dict[str, Any]:
        """Evaluate on QCM dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        if dataset_path:
            self.dataset_path = Path(dataset_path)
        if image_dir:
            self.image_dir = Path(image_dir)

        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError(f"QCM dataset not found: {self.dataset_path}")

        logger.info(f"Evaluating on QCM dataset: {self.dataset_path}")

        # Load QCM dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle nested structure if present
        if raw_data and 'qcm' in raw_data[0]:
            dataset = [(item['qcm'], item.get('image_name', '')) for item in raw_data]
        else:
            dataset = [(item, item.get('image_name', '')) for item in raw_data]

        if max_samples:
            dataset = dataset[:max_samples]

        results = []
        for qcm_data, image_name in tqdm(dataset, desc="QCM Evaluation"):
            # Load image
            if image_name and self.image_dir:
                image_path = self.image_dir / image_name
                if image_path.exists():
                    image = self.load_image(str(image_path))
                else:
                    image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.new('RGB', (224, 224), color='white')

            # Format question with options
            question = qcm_data['question']
            options = qcm_data['options']
            correct_answer = qcm_data['correct_answer']

            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = f"{question}\n\nOptions:\n{options_text}\n\nFirst, state the letter of the correct answer. YOU MUST OUTPUT THE CORRECT LETTER FIRST, then the text of the answer, then provide your explanation.\n\nAnswer:"

            response = self.generate_response(image, prompt)

            # Extract predicted answer (letter)
            predicted_letter = self._extract_answer_letter(response, list(options.keys()))

            results.append({
                "question": question,
                "options": options,
                "response": response,
                "predicted_letter": predicted_letter,
                "correct_answer": correct_answer,
                "is_correct": predicted_letter == correct_answer,
                "dataset": "erp_qcm"
            })

        accuracy = self.calculate_accuracy(results)

        return {
            "benchmark": "erp_qcm",
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct": sum(1 for r in results if r['is_correct']),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate QCM accuracy with lenient matching"""
        if not results:
            return 0.0

        correct = 0
        total = len(results)

        for r in results:
            # First check if letter matches (strict)
            if r.get('is_correct', False):
                correct += 1
                continue

            # Lenient: check if correct option text is in response (or vice versa)
            if 'options' in r and 'correct_answer' in r and 'response' in r:
                correct_letter = r['correct_answer']
                options = r['options']
                response = r['response']

                if correct_letter in options:
                    correct_text = self._normalize_text(options[correct_letter])
                    response_norm = self._normalize_text(response)

                    # Check if correct option text is in response or vice versa
                    if correct_text and response_norm:
                        if correct_text in response_norm or response_norm in correct_text:
                            correct += 1

        return (correct / total * 100) if total > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """Normalize text for lenient comparison"""
        text = str(text).lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove all whitespace
        text = re.sub(r'\s+', '', text)
        return text
        
    def _extract_answer_letter(self, response: str, valid_options: List[str]) -> str:
        """Extract the answer letter from the response"""
        response = response.strip().upper()
        valid_upper = [o.upper() for o in valid_options]
        
        # Sort by length descending so "AA" is matched before "A" in regex alternation
        options_pattern = '|'.join(sorted(valid_upper, key=len, reverse=True))
        
        # Look for common answer patterns (ordered by specificity)
        patterns = [
            # === ENGLISH PATTERNS ===
            
            # Final/definitive answers (highest priority for chain-of-thought)
            rf'FINAL\s+ANSWER\s+IS\s+({options_pattern})',
            rf'FINAL\s+ANSWER[:\s]+({options_pattern})',
            rf'FINAL\s+CHOICE\s+IS\s+({options_pattern})',
            rf'(?:THE\s+)?(?:CORRECT|RIGHT|BEST)\s+(?:ANSWER|OPTION|CHOICE)\s+IS\s+({options_pattern})',
            rf'(?:THE\s+)?CORRECT\s+ANSWER\s+IS\s+({options_pattern})',
            rf'CORRECT[:\s]+({options_pattern})',
            rf'CORRECT\s+(?:OPTION|CHOICE|ANSWER)[:\s]+({options_pattern})',
            
            # Standard answer patterns
            rf'(?:THE\s+)?ANSWER\s+IS\s+({options_pattern})',
            rf'(?:THE\s+)?RESPONSE\s+IS\s+({options_pattern})',
            rf'(?:THE\s+)?ANSWER[:\s]+({options_pattern})',
            rf'(?:THE\s+)?RESPONSE[:\s]+({options_pattern})',
            rf'ANSWER\s*=\s*({options_pattern})',
            
            # Personal choice patterns
            rf'MY\s+(?:ANSWER|CHOICE|RESPONSE)\s+IS\s+({options_pattern})',
            rf'I\s+(?:WOULD\s+)?(?:CHOOSE|SELECT|PICK)\s+({options_pattern})',
            rf'I\s+(?:BELIEVE|THINK)\s+(?:IT\'?S?|THE\s+ANSWER\s+IS)\s+({options_pattern})',
            rf'(?:I\'?LL?\s+)?GO(?:ING)?\s+WITH\s+({options_pattern})',
            
            # Modal patterns
            rf'SHOULD\s+BE\s+({options_pattern})',
            rf'WOULD\s+BE\s+({options_pattern})',
            rf'MUST\s+BE\s+({options_pattern})',
            rf'IT\'?S?\s+({options_pattern})',
            
            # Reverse patterns ("A is correct")
            rf'({options_pattern})\s+IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|BEST|ANSWER)',
            
            # Conclusion indicators
            rf'(?:THEREFORE|THUS|HENCE|SO)[,:\s]+({options_pattern})',
            
            # Labeled patterns
            rf'IS\s+({options_pattern})',
            rf'OPTION\s+({options_pattern})',
            rf'CHOICE\s+({options_pattern})',
            rf'SELECT\s+({options_pattern})',
            rf'CHOOSE\s+({options_pattern})',
            
            # === FRENCH PATTERNS ===
            
            # Final/definitive answers
            rf'R[ÉE]PONSE\s+FINALE\s+EST\s+({options_pattern})',      # "réponse finale est A"
            rf'R[ÉE]PONSE\s+FINALE[:\s]+({options_pattern})',         # "réponse finale: A"
            rf'CHOIX\s+FINAL\s+EST\s+({options_pattern})',            # "choix final est A"
            rf'CHOIX\s+FINAL[:\s]+({options_pattern})',               # "choix final: A"
            rf'(?:LA\s+)?(?:BONNE|CORRECTE)\s+R[ÉE]PONSE\s+EST\s+({options_pattern})',  # "la bonne réponse est A"
            rf'(?:LA\s+)?R[ÉE]PONSE\s+(?:CORRECTE|BONNE)\s+EST\s+({options_pattern})',  # "la réponse correcte est A"
            rf'CORRECT[:\s]+({options_pattern})',
            
            # Standard answer patterns
            rf'(?:LA\s+)?R[ÉE]PONSE\s+EST\s+({options_pattern})',     # "la réponse est A" / "réponse est A"
            rf'(?:LA\s+)?R[ÉE]PONSE[:\s]+({options_pattern})',        # "réponse: A" / "réponse A"
            rf'R[ÉE]PONSE\s*=\s*({options_pattern})',                 # "réponse = A"
            
            # Personal choice patterns
            rf'MA\s+R[ÉE]PONSE\s+EST\s+({options_pattern})',          # "ma réponse est A"
            rf'MON\s+CHOIX\s+EST\s+({options_pattern})',              # "mon choix est A"
            rf'JE\s+(?:CHOISIS|S[ÉE]LECTIONNE)\s+({options_pattern})', # "je choisis A" / "je sélectionne A"
            rf'J\'?OPTERAIS?\s+POUR\s+({options_pattern})',           # "j'opterais pour A" / "j'opte pour A"
            rf'JE\s+(?:PENSE|CROIS)\s+QUE\s+C\'?EST\s+({options_pattern})',  # "je pense que c'est A"
            rf'JE\s+(?:PENSE|CROIS)\s+QUE\s+LA\s+R[ÉE]PONSE\s+EST\s+({options_pattern})',  # "je pense que la réponse est A"
            rf'JE\s+(?:DIRAIS?|PENCHERAIS?)\s+POUR\s+({options_pattern})',  # "je dirais A" / "je pencherais pour A"
            
            # Modal patterns
            rf'(?:CE\s+)?DEVRAIT\s+[ÊE]TRE\s+({options_pattern})',    # "ce devrait être A" / "devrait être A"
            rf'(?:CE\s+)?SERAIT\s+({options_pattern})',               # "ce serait A"
            rf'(?:CE\s+)?DOIT\s+[ÊE]TRE\s+({options_pattern})',       # "ce doit être A"
            rf'C\'?EST\s+({options_pattern})',                        # "c'est A"
            rf'IL\s+S\'?AGIT\s+DE\s+({options_pattern})',             # "il s'agit de A"
            
            # Reverse patterns ("A est correct")
            rf'({options_pattern})\s+EST\s+(?:LA\s+)?(?:BONNE|CORRECTE)\s+R[ÉE]PONSE',  # "A est la bonne réponse"
            rf'({options_pattern})\s+EST\s+(?:CORRECT|JUSTE|BON)',    # "A est correct"
            
            # Conclusion indicators
            rf'(?:DONC|AINSI|PAR\s+CONS[ÉE]QUENT|EN\s+CONCLUSION)[,:\s]+({options_pattern})',  # "donc, A"
            rf'(?:FINALEMENT|POUR\s+CONCLURE)[,:\s]+({options_pattern})',  # "finalement, A"
            
            # Labeled patterns
            rf'EST\s+({options_pattern})',                            # "est A"
            rf'OPTION\s+({options_pattern})',                         # "option A"
            rf'CHOIX\s+({options_pattern})',                          # "choix A"
            rf'S[ÉE]LECTIONNER\s+({options_pattern})',               # "sélectionner A"
            rf'CHOISIR\s+({options_pattern})',                        # "choisir A"
            
            # === FORMATTING PATTERNS (language-agnostic) ===
            
            rf'\*\*({options_pattern})\*\*',                          # **A** or **AA**
            rf'\*({options_pattern})\*',                              # *A* or *AA*
            rf'`({options_pattern})`',                                # `A` or `AA`
            rf'\(({options_pattern})\)',                              # (A) or (AA)
            rf'({options_pattern})\)',                                # A) or AA)
            rf'({options_pattern}):',                                 # A: or AA:
            rf'({options_pattern})\s*-',                              # A - or AA-
            
            # Bullet point patterns
            rf'^[\-\*\•]\s*({options_pattern})',                      # "- A" / "* A" / "• A"
            rf'[\-\*\•]\s*({options_pattern})',                       # same but not at start
            
            # Generic match (lowest priority)
            rf'({options_pattern})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1)
        
        return ""