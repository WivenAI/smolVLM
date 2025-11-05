#!/bin/bash
# Automatically run sanity check when training completes

echo "==========================================================================="
echo "Monitoring training completion..."
echo "==========================================================================="

# Wait for training to complete (check if process is still running)
while pgrep -f "finetune_on_benchmarks.py --benchmark ocrbench --num-epochs 1" > /dev/null; do
    echo "Training still running... ($(date +%H:%M:%S))"
    sleep 30
done

echo ""
echo "✅ Training completed!"
echo ""

# Check if model was saved
if [ ! -d "test_ocrbench_1epoch" ]; then
    echo "❌ Error: Model directory not found"
    exit 1
fi

echo "==========================================================================="
echo "STEP 1: Evaluating BASE MODEL on 1000 OCRBench samples"
echo "==========================================================================="
python3 evaluate_ocrbench.py \
    --percentage 100 \
    --output-dir sanity_check_results/base_model

echo ""
echo "==========================================================================="
echo "STEP 2: Evaluating TRAINED MODEL on 1000 OCRBench samples"
echo "==========================================================================="
python3 evaluate_ocrbench.py \
    --model-path test_ocrbench_1epoch \
    --percentage 100 \
    --output-dir sanity_check_results/trained_model

echo ""
echo "==========================================================================="
echo "STEP 3: Running SANITY CHECK - Comparing Base vs Trained"
echo "==========================================================================="

# Find the most recent result files
BASE_RESULTS=$(ls -t sanity_check_results/base_model/*.json 2>/dev/null | head -1)
TRAINED_RESULTS=$(ls -t sanity_check_results/trained_model/*.json 2>/dev/null | head -1)

if [ -z "$BASE_RESULTS" ] || [ -z "$TRAINED_RESULTS" ]; then
    echo "❌ Error: Could not find result files"
    echo "Base: $BASE_RESULTS"
    echo "Trained: $TRAINED_RESULTS"
    exit 1
fi

python3 sanity_check_training.py \
    --base-results "$BASE_RESULTS" \
    --trained-results "$TRAINED_RESULTS"

echo ""
echo "==========================================================================="
echo "SANITY CHECK COMPLETE!"
echo "==========================================================================="
echo ""
echo "Result files:"
echo "  Base:    $BASE_RESULTS"
echo "  Trained: $TRAINED_RESULTS"
echo ""
