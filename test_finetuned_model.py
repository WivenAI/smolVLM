#!/usr/bin/env python3
"""
Test script for the fine-tuned SmolVLM model
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from io import BytesIO

def load_finetuned_model(model_path="./smolvlm-finetuned"):
    """Load the fine-tuned model and processor"""
    print(f"Loading fine-tuned model from: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    return model, processor

def test_model_with_image(model, processor, image_path_or_url, question):
    """Test the model with an image and question"""

    # Load image
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    # Use same format as training
    text = f"<image>Question: {question}\nAnswer:"

    # Process inputs
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # Decode response
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the assistant's response
    if "assistant" in generated_text:
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text

    return response

def main():
    """Main testing function"""
    print("Testing fine-tuned SmolVLM model...")

    try:
        # Load the fine-tuned model
        model, processor = load_finetuned_model()

        # Test with a sample image URL
        test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        test_questions = [
            "What is in the image?",
            "What color is the car?",
            "Is this vehicle suitable for off-road driving?"
        ]

        print("\n" + "="*50)
        print("TESTING FINE-TUNED MODEL")
        print("="*50)

        for i, question in enumerate(test_questions, 1):
            print(f"\nTest {i}:")
            print(f"Question: {question}")

            try:
                answer = test_model_with_image(model, processor, test_image_url, question)
                print(f"Answer: {answer}")
            except Exception as e:
                print(f"Error generating answer: {e}")

            print("-" * 30)

        print("\nTesting completed!")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run the fine-tuning script first.")

if __name__ == "__main__":
    main()