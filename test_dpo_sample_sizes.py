#!/usr/bin/env python3
"""
Test DPO training with different sample sizes to find optimal size
"""

import subprocess
import time
import sys

def test_dpo_with_samples(num_samples, timeout=300):
    """Test DPO training with specific number of samples"""

    # First, modify the max_samples in the script temporarily
    with open('finetune_smolvlm_dpo.py', 'r') as f:
        content = f.read()

    # Replace the max_samples line
    modified = content.replace(
        'max_samples = 500  # Use 500 samples max for DPO training',
        f'max_samples = {num_samples}  # Testing with {num_samples} samples'
    )

    with open('finetune_smolvlm_dpo.py', 'w') as f:
        f.write(modified)

    print(f"\n{'='*80}")
    print(f"Testing DPO with {num_samples} samples")
    print(f"{'='*80}\n")

    try:
        # Run DPO training
        result = subprocess.run(
            [
                'python3', 'finetune_smolvlm_dpo.py',
                '--dataset', 'dpo_image_dataset/dpo_dataset_gemini.json',
                '--image-dir', 'dpo_image_dataset',
                '--output-dir', f'test_dpo_{num_samples}samples'
            ],
            timeout=timeout,
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr

        # Check for success indicators
        if 'Tokenizing train dataset: 100%' in output or 'Starting DPO training' in output:
            print(f"✅ SUCCESS with {num_samples} samples!")
            print(f"   Tokenization completed successfully")
            return True, "Success"
        elif 'CUDA out of memory' in output or 'OOM' in output:
            print(f"❌ FAILED with {num_samples} samples - OOM")
            return False, "OOM"
        elif result.returncode != 0:
            print(f"❌ FAILED with {num_samples} samples - Error")
            # Print last 50 lines of output
            lines = output.split('\n')
            print('\n'.join(lines[-50:]))
            return False, "Error"
        else:
            print(f"⚠️  UNKNOWN with {num_samples} samples")
            return False, "Unknown"

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT with {num_samples} samples (>{timeout}s)")
        return False, "Timeout"

    finally:
        # Restore original file
        with open('finetune_smolvlm_dpo.py', 'w') as f:
            f.write(content)

def main():
    # Test different sample sizes
    test_sizes = [
        (100, 180),   # Very safe
        (250, 240),   # Conservative
        (500, 300),   # Current setting
        (750, 360),   # Aggressive (if you want to try)
    ]

    results = []

    print("="*80)
    print("DPO SAMPLE SIZE TESTING")
    print("="*80)
    print("\nTesting different sample sizes to find optimal configuration")
    print("This will test tokenization phase (most memory intensive)")
    print("\nPress Ctrl+C to stop testing\n")

    for num_samples, timeout in test_sizes:
        try:
            success, status = test_dpo_with_samples(num_samples, timeout)
            results.append({
                'samples': num_samples,
                'success': success,
                'status': status
            })

            # If we hit OOM, no point testing larger sizes
            if status == "OOM":
                print(f"\n⚠️  Hit OOM at {num_samples} samples, stopping tests")
                break

            # If successful, we can try next size
            if success:
                print(f"\n✅ {num_samples} samples works! Trying next size...")
                time.sleep(5)  # Brief pause between tests
            else:
                break

        except KeyboardInterrupt:
            print("\n\n⚠️  Testing interrupted by user")
            break

    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    for result in results:
        status_symbol = "✅" if result['success'] else "❌"
        print(f"{status_symbol} {result['samples']:4d} samples: {result['status']}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    successful = [r for r in results if r['success']]
    if successful:
        max_working = max(r['samples'] for r in successful)
        print(f"\n✅ Maximum working size: {max_working} samples")
        print(f"\nUpdate finetune_smolvlm_dpo.py:")
        print(f"   max_samples = {max_working}")
    else:
        print("\n❌ No configuration worked!")
        print("   Try reducing to 50 or 100 samples")
        print("   Or skip DPO training with --skip-erp-dpo")

if __name__ == '__main__':
    main()
