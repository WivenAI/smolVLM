"""
Diagnose if ChartQA answer positions are being found correctly
"""
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from config.setup import setup_hf_cache, BASE_MODEL

setup_hf_cache()

CACHE_DIR = Path("datasets/cache")
CHARTQA_CACHE = CACHE_DIR / "HuggingFaceM4_ChartQA_test.json"

def diagnose_masking():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading ChartQA from {CHARTQA_CACHE}...")
    with open(CHARTQA_CACHE, 'r') as f:
        data = json.load(f)

    # Check first 50 samples
    samples_to_check = data[:50]

    fallback_count = 0
    found_count = 0
    issues = []

    for idx, item in enumerate(samples_to_check):
        question = item.get('query', '')
        answer = item.get('label', [''])[0] if isinstance(item.get('label'), list) else item.get('label', '')
        image_path = item.get('image_path', '')

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            issues.append((idx, "Image load failed", answer))
            continue

        # Create full messages (same as trainer)
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        full_inputs = processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Try to find answer position (FIXED: try with leading space first)
        full_token_list = full_inputs["input_ids"][0].tolist()
        answer_start_pos = None

        # Try 1: Answer with leading space (most common in chat templates)
        answer_with_space = " " + answer
        answer_tokens_spaced = processor.tokenizer.encode(answer_with_space, add_special_tokens=False)
        for i in range(len(full_token_list) - len(answer_tokens_spaced) + 1):
            if full_token_list[i:i+len(answer_tokens_spaced)] == answer_tokens_spaced:
                answer_start_pos = i
                break

        # Try 2: Answer without leading space
        if answer_start_pos is None:
            answer_tokens = processor.tokenizer.encode(answer, add_special_tokens=False)
            for i in range(len(full_token_list) - len(answer_tokens) + 1):
                if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                    answer_start_pos = i
                    break

        # Check if fallback would be used
        if answer_start_pos is None:
            fallback_count += 1

            # Try "Assistant:" marker fallback
            assistant_found = False
            try:
                assistant_tokens = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
                for i in range(len(full_token_list) - len(assistant_tokens) + 1):
                    if full_token_list[i:i+len(assistant_tokens)] == assistant_tokens:
                        assistant_found = True
                        break
            except:
                pass

            issues.append({
                'idx': idx,
                'answer': answer,
                'answer_tokens': answer_tokens,
                'total_tokens': len(full_token_list),
                'assistant_marker_found': assistant_found,
                'decoded_answer_tokens': processor.tokenizer.decode(answer_tokens)
            })
        else:
            found_count += 1

            # Calculate what percentage of sequence is unmasked
            unmasked_pct = (len(full_token_list) - answer_start_pos) / len(full_token_list) * 100

            if unmasked_pct > 50:  # If more than 50% unmasked, something might be wrong
                issues.append({
                    'idx': idx,
                    'answer': answer,
                    'answer_start_pos': answer_start_pos,
                    'total_tokens': len(full_token_list),
                    'unmasked_pct': unmasked_pct,
                    'issue': 'Too much unmasked - answer might be in wrong position'
                })

    print("\n" + "="*80)
    print("DIAGNOSIS RESULTS")
    print("="*80)
    print(f"Samples checked: {len(samples_to_check)}")
    print(f"Answer position FOUND: {found_count}")
    print(f"Answer position NOT FOUND (fallback used): {fallback_count}")
    print(f"Fallback rate: {fallback_count/len(samples_to_check)*100:.1f}%")

    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        print("\nFirst 10 issues:")
        for i, issue in enumerate(issues[:10]):
            print(f"\n--- Issue {i+1} ---")
            for k, v in issue.items() if isinstance(issue, dict) else [('issue', issue)]:
                print(f"  {k}: {v}")

    # Show some example answers that failed
    if fallback_count > 0:
        print("\n" + "="*80)
        print("SAMPLE ANSWERS THAT COULDN'T BE FOUND:")
        print("="*80)
        for issue in [i for i in issues if 'answer_tokens' in i][:5]:
            print(f"\nSample {issue['idx']}:")
            print(f"  Answer text: '{issue['answer']}'")
            print(f"  Answer tokens: {issue['answer_tokens']}")
            print(f"  Decoded tokens: '{issue['decoded_answer_tokens']}'")

if __name__ == "__main__":
    diagnose_masking()
