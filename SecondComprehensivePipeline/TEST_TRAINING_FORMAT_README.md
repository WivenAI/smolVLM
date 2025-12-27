# Training Format Test Script

This script (`test_training_format.py`) helps you visualize exactly how your training data is formatted and what the model is being trained on.

## What It Shows

For each sample, the script displays:

1. **Original Data** - The raw QCM question/DPO preference pair
2. **Formatted Messages** - How the data is structured as user/assistant messages
3. **Chat Template Applied** - The actual text after applying the chat template
4. **Tokenization** - Token counts for prompt and response
5. **Labels (Masking)** - Which tokens are masked (-100) vs trained
6. **Token-by-Token Breakdown** - Individual tokens with their labels
7. **Training Target** - What the model is actually trained to predict

## Usage

### Test QCM Training Format

```bash
# Gemini QCM dataset
python test_training_format.py --dataset qcm_gemini --num-samples 3

# Nova QCM dataset
python test_training_format.py --dataset qcm_nova --num-samples 3

# Procedure datasets
python test_training_format.py --dataset qcm_procedure1 --num-samples 3
python test_training_format.py --dataset qcm_procedure2 --num-samples 3
```

### Test DPO Training Format

```bash
# Gemini DPO dataset
python test_training_format.py --dataset dpo_gemini --num-samples 3

# Nova DPO dataset
python test_training_format.py --dataset dpo_nova --num-samples 3
```

### Options

- `--dataset` - Dataset to test (required)
  - QCM: `qcm_gemini`, `qcm_nova`, `qcm_procedure1`, `qcm_procedure2`
  - DPO: `dpo_gemini`, `dpo_nova`
- `--num-samples` - Number of samples to visualize (default: 3)

## Example Output

### QCM Training Format

```
================================================================================
SAMPLE 1
================================================================================

[1] ORIGINAL DATA:
--------------------------------------------------------------------------------
Question: What is the main menu item?
Options: {
  "A": "Sales",
  "B": "Purchase",
  "C": "Inventory",
  "D": "Reports"
}
Correct Answer: A
Image: screenshot_001.jpg

[2] FORMATTED MESSAGES:
--------------------------------------------------------------------------------
User Message:
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Answer briefly."},
      {"type": "image"},
      {"type": "text", "text": "What is the main menu item?\n\nOptions:\nA: Sales\nB: Purchase\nC: Inventory\nD: Reports\n\n..."}
    ]
  }
]

[3] CHAT TEMPLATE APPLIED:
--------------------------------------------------------------------------------
Prompt Text (what model sees as input):
'<|im_start|>user\nAnswer briefly.<image>What is the main menu item?...<|im_end|>\n<|im_start|>assistant\n'

Full Text (prompt + response for training):
'<|im_start|>user\nAnswer briefly.<image>What is the main menu item?...<|im_end|>\n<|im_start|>assistant\nA<|im_end|>'

[4] TOKENIZATION:
--------------------------------------------------------------------------------
Prompt token length: 124
Full sequence token length: 126
Response token length: 2

[5] LABELS (MASKING):
--------------------------------------------------------------------------------
Masked tokens (set to -100): 124 tokens
Trainable tokens: 2 tokens

Label masking visualization:
  Prompt tokens (0 to 123): MASKED with -100 (not trained)
  Response tokens (124 to 125): TRAINED

[6] TOKEN-BY-TOKEN BREAKDOWN:
--------------------------------------------------------------------------------
Token ID   Token                          Label      Trained?
----------------------------------------------------------------------
151644     '<|im_start|>'                 -100       NO (masked)
882        'user'                         -100       NO (masked)
198        '\n'                           -100       NO (masked)
16141      'Answer'                       -100       NO (masked)
...

Last 10 tokens (should be the response being trained):
Token ID   Token                          Label      Trained?
----------------------------------------------------------------------
151645     '<|im_end|>'                   -100       NO (masked)
198        '\n'                           -100       NO (masked)
151644     '<|im_start|>'                 -100       NO (masked)
78191      'assistant'                    -100       NO (masked)
198        '\n'                           -100       NO (masked)
32         'A'                            32         YES
151645     '<|im_end|>'                   151645     YES

[7] TRAINING TARGET:
--------------------------------------------------------------------------------
Model is trained to output: 'A'
Expected output: 'A'
```

### DPO Training Format

```
================================================================================
SAMPLE 1
================================================================================

[1] ORIGINAL DATA:
--------------------------------------------------------------------------------
Prompt: What field is being edited in the form?
Chosen Response: The "Customer Name" field is being edited in the customer information form.
Rejected Response: A text field is being edited.
Image: screenshot_002.jpg

[2] FORMATTED MESSAGES (CHOSEN):
--------------------------------------------------------------------------------
User Message:
[
  {
    "role": "user",
    "content": [
      {"type": "image"},
      {"type": "text", "text": "What field is being edited in the form?"}
    ]
  }
]

Chosen Messages:
[
  ... with chosen response ...
]

Rejected Messages:
[
  ... with rejected response ...
]

[4] DPO TRAINING:
--------------------------------------------------------------------------------
DPO trains the model to:
  1. Increase log probability of CHOSEN response
  2. Decrease log probability of REJECTED response
  3. Maximize the margin: log P(chosen) - log P(rejected)

The model learns to prefer the chosen response over the rejected one
by optimizing a preference-based loss function.

[6] TRAINING TARGET:
--------------------------------------------------------------------------------
Model is trained to:
  PREFER (increase probability): 'The "Customer Name" field is being edited...'
  REJECT (decrease probability): 'A text field is being edited.'
```

## Understanding the Output

### Key Concepts

1. **Masking with -100**: Tokens set to -100 in the labels are ignored during loss calculation. This means the model is NOT trained to predict these tokens. Typically, the entire prompt is masked.

2. **Trainable Tokens**: Only the response tokens (after the prompt) have real label values and contribute to the training loss.

3. **Chat Template**: The processor applies a chat template that adds special tokens like `<|im_start|>`, `<|im_end|>`, etc. Understanding this template is crucial for understanding what the model sees.

4. **Token IDs**: Each token is represented by a numerical ID. The model learns to predict these IDs.

### For QCM Training

- The model is trained to output just the letter (A, B, C, or D)
- The entire question and options are in the prompt (masked)
- Only the answer letter is trained

### For DPO Training

- The model sees both chosen and rejected responses
- It learns to increase probability of chosen response
- It learns to decrease probability of rejected response
- The margin between them is maximized

## Troubleshooting

### Image Not Found

If images are not found, the script will use blank white images. This won't affect the text formatting visualization.

### Dataset Not Found

Make sure you're running from the SecondComprehensivePipeline directory and the dataset paths are correct.

## Use Cases

1. **Debug Training Issues**: See if the data is formatted correctly
2. **Understand Masking**: Verify which tokens are being trained
3. **Check Chat Template**: Ensure the chat template is applied correctly
4. **Token Count Verification**: Check if sequences are too long
5. **Compare Datasets**: See differences between QCM and DPO formatting
