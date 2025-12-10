# DPO Dataset Generator with CrewAI

This project uses CrewAI agents and Google Gemini Vision API to generate DPO (Direct Preference Optimization) datasets from images and their corresponding context text files.

## Overview

The system analyzes images (PNG) along with their context files (TXT) and generates:
- **Descriptive prompts**: Natural questions asking about image content
- **Q&A prompts**: Realistic user questions about the software features shown
- **Chosen responses**: High-quality, detailed answers (preferred)
- **Rejected responses**: Lower-quality, vague answers (not preferred)

Each image generates 2 DPO entries (descriptive + Q&A).

## Files

- `generate_dpo_dataset.py`: Main script to process all images in the directory
- `test_single_image.py`: Test script for a single image (image_001.png)
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Requirements

```bash
pip install crewai google-generativeai pillow
```

## Setup

1. Ensure your Gemini API key is stored at: `/home/david-lacour/Desktop/geminiAPIkey.txt`
2. Images should be named: `image_XXX.png`
3. Context files should have the same name: `image_XXX.txt`

## Usage

### Test with a single image:
```bash
python3 test_single_image.py
```

This will:
- Process `image_001.png` and `image_001.txt`
- Generate DPO entries
- Save output to `dpo_test_output.json`

### Generate full dataset:
```bash
python3 generate_dpo_dataset.py
```

This will:
- Process all image/text pairs in the directory
- Generate 2 DPO entries per image
- **Save progress after each image** (automatic checkpointing)
- Save the complete dataset to `dpo_dataset.json`

**Resume capability**: If the script is interrupted, simply run it again. It will:
- Load the existing `dpo_dataset.json`
- Skip already processed images
- Continue from where it left off

## Output Format

Each DPO entry contains:
```json
{
  "prompt": "User's question about the image",
  "chosen": "High-quality detailed response",
  "rejected": "Lower-quality vague response",
  "image_name": "image_001.png",
  "type": "descriptive" or "qa"
}
```

## CrewAI Agents

The system uses 2 specialized agents:

1. **Prompt Generation Agent**: Creates realistic user prompts
2. **Response Generation Agent**: Generates chosen/rejected response pairs

## How It Works

1. **Image Analysis**: Gemini Vision API analyzes each image with its context
2. **Prompt Generation**: Agents create natural user questions
3. **Response Generation**: Agents create high-quality (chosen) and low-quality (rejected) responses
4. **Dataset Creation**: All entries are compiled into a JSON file

## Example Output

```json
{
  "prompt": "What is the 'Webconfig - configuration MSWeb' window used for?",
  "chosen": "The Webconfig window is a dedicated configuration tool for...",
  "rejected": "The Webconfig window is used to configure parts of MSWeb...",
  "image_name": "image_001.png",
  "type": "qa"
}
```

## Features

✅ **Automatic Progress Saving**: Dataset is saved after each image is processed
✅ **Resume Capability**: Restart the script anytime - it skips completed images
✅ **Error Handling**: Continues processing even if individual images fail
✅ **Progress Tracking**: Shows current progress and total entries

## Notes

- The system uses `gemini-2.5-flash` model
- Processing time: ~30-60 seconds per image
- Images must be in PNG format
- Context files must be UTF-8 encoded text
- **Safe to interrupt**: Press Ctrl+C anytime, progress is saved
- Run again to resume from last completed image
