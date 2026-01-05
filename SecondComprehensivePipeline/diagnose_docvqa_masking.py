"""
Diagnose if DocVQA answer positions are being found correctly
"""
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from config.setup import setup_hf_cache, BASE_MODEL

setup_hf_cache()

CACHE_DIR = Path("datasets/cache")
DOCVQA_CACHE = CACHE_DIR / "nielsr_docvqa_1200_examples_train.json"

def diagnose_masking():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading DocVQA from {DOCVQA_CACHE}...")
    with open(DOCVQA_CACHE, 'r') as f:
        data = json.load(f)

    # Check first 50 samples
    samples_to_check = data[:50]

    fallback_count = 0
    found_count = 0
    issues = []

    for idx, item in enumerate(samples_to_check):
        # Handle query field which can be a dict with language keys
        query_field = item.get('query', '')
        if isinstance(query_field, dict):
            question = query_field.get('en', str(query_field))
        else:
            question = str(query_field)

        # Get first answer
        answers = item.get('answers', [])
        answer = answers[0] if answers else ''

        image_path = item.get('image_path', '')

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            issues.append({'idx': idx, 'error': f'Image load failed: {e}', 'answer': answer})
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
            assistant_pos = None
            try:
                for marker in ["Assistant:", "Assistant: ", ": "]:
                    marker_tokens = processor.tokenizer.encode(marker, add_special_tokens=False)
                    for i in range(len(full_token_list) - len(marker_tokens) + 1):
                        if full_token_list[i:i+len(marker_tokens)] == marker_tokens:
                            assistant_found = True
                            assistant_pos = i + len(marker_tokens)
                            break
                    if assistant_found:
                        break
            except:
                pass

            issues.append({
                'idx': idx,
                'answer': answer,
                'answer_tokens_spaced': processor.tokenizer.encode(' ' + answer, add_special_tokens=False),
                'answer_tokens_no_space': processor.tokenizer.encode(answer, add_special_tokens=False),
                'total_tokens': len(full_token_list),
                'assistant_marker_found': assistant_found,
                'assistant_pos': assistant_pos,
                'decoded_spaced': processor.tokenizer.decode(processor.tokenizer.encode(' ' + answer, add_special_tokens=False)),
                'last_10_tokens': full_token_list[-10:],
                'decoded_last_10': [processor.tokenizer.decode([t]) for t in full_token_list[-10:]]
            })
        else:
            found_count += 1

    print("\n" + "="*80)
    print("DIAGNOSIS RESULTS")
    print("="*80)
    print(f"Samples checked: {len(samples_to_check)}")
    print(f"Answer position FOUND: {found_count}")
    print(f"Answer position NOT FOUND (would crash): {fallback_count}")
    print(f"Failure rate: {fallback_count/len(samples_to_check)*100:.1f}%")

    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        print("\nFirst 10 issues:")
        for i, issue in enumerate(issues[:10]):
            print(f"\n--- Issue {i+1} ---")
            for k, v in issue.items():
                print(f"  {k}: {v}")

    # Show tokenization examples
    if fallback_count > 0:
        print("\n" + "="*80)
        print("SAMPLE ANSWERS THAT COULDN'T BE FOUND:")
        print("="*80)
        for issue in [i for i in issues if 'answer_tokens_spaced' in i][:5]:
            print(f"\nSample {issue['idx']}:")
            print(f"  Answer text: '{issue['answer']}'")
            print(f"  With space tokens: {issue['answer_tokens_spaced']}")
            print(f"  No space tokens: {issue['answer_tokens_no_space']}")
            print(f"  Decoded (spaced): '{issue['decoded_spaced']}'")
            print(f"  Last 10 decoded: {issue['decoded_last_10']}")

if __name__ == "__main__":
    diagnose_masking()
