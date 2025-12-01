#!/usr/bin/env python3
"""
Test script to generate DPO dataset for a single image
"""

import os
import json
from pathlib import Path
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process, LLM

# Configure Gemini API
with open('/home/david-lacour/Desktop/geminiAPIkey.txt', 'r') as f:
    GEMINI_API_KEY = f.read().strip()

genai.configure(api_key=GEMINI_API_KEY)

# Configure LLM for CrewAI
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

WORKING_DIR = "/home/david-lacour/Documents/Wiven/data/exctractWithContext/gemini/output_same_folder_without_icone"

# Test with first image
image_path = os.path.join(WORKING_DIR, "image_001.png")
text_path = os.path.join(WORKING_DIR, "image_001.txt")

print(f"Testing with: {image_path}")

# Read context
with open(text_path, 'r', encoding='utf-8') as f:
    context = f.read()

# Initialize Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

# Load and analyze image
print("Loading and analyzing image...")
from PIL import Image

image = Image.open(image_path)

prompt = f"""You are analyzing an image from a software documentation system.

Context from the documentation:
{context}

Please analyze this image and provide:
1. A detailed description of what the image shows
2. How it relates to the provided context
3. What specific UI elements, buttons, or features are visible
4. Any text or labels visible in the image

Be specific and technical in your analysis."""

response = model.generate_content([prompt, image])
image_analysis = response.text

print("\nImage Analysis:")
print("=" * 60)
print(image_analysis)
print("=" * 60)

# Create agents
prompt_generation_agent = Agent(
    role='Training Data Prompt Engineer',
    goal='Generate high-quality prompts for DPO dataset creation',
    backstory="""You are an expert in creating training datasets for vision-language models.
    You excel at crafting diverse, realistic prompts that users might ask when interacting
    with a visual AI assistant about software documentation.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

response_generation_agent = Agent(
    role='Assistant Response Generator',
    goal='Generate chosen (high-quality) and rejected (lower-quality) responses for DPO training',
    backstory="""You are an expert at generating pairs of responses for preference learning.
    You understand what makes a good assistant response (accurate, helpful, well-structured)
    versus a poor one (vague, incomplete, or incorrect).""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Create tasks
task1 = Task(
    description=f"""Based on this image analysis and context, generate a descriptive prompt
    that asks the AI to describe what's in the image.

    Image Analysis: {image_analysis[:500]}...
    Context: {context[:500]}...

    Generate a natural user prompt asking about the image content.
    Respond with ONLY the prompt, nothing else.""",
    agent=prompt_generation_agent,
    expected_output="A natural language prompt asking about the image"
)

task2 = Task(
    description=f"""Based on this image analysis and context, generate a question that a user
    might ask about this software feature or UI element shown in the image.

    Image Analysis: {image_analysis[:500]}...
    Context: {context[:500]}...

    Generate a realistic user question about the functionality shown.
    Respond with ONLY the question, nothing else.""",
    agent=prompt_generation_agent,
    expected_output="A realistic user question about the software feature"
)

task3 = Task(
    description=f"""Generate a high-quality assistant response that:
    1. Accurately describes the image
    2. References the relevant context
    3. Is helpful, clear, and well-structured

    Image Analysis: {image_analysis}
    Context: {context}

    This should be the 'chosen' response - the preferred, high-quality answer.
    Write as if you're an AI assistant helping a user understand this software.""",
    agent=response_generation_agent,
    expected_output="A high-quality, detailed assistant response"
)

task4 = Task(
    description=f"""Generate a lower-quality assistant response that:
    1. Is vague or incomplete
    2. Misses important details from the image
    3. Doesn't fully address the question
    4. Is less structured and helpful

    Image Analysis: {image_analysis}
    Context: {context}

    This should be the 'rejected' response - a less helpful answer.
    Write as if you're an AI assistant giving a mediocre response.""",
    agent=response_generation_agent,
    expected_output="A lower-quality, vague assistant response"
)

# Create and run crew
print("\nGenerating prompts and responses...")
crew = Crew(
    agents=[prompt_generation_agent, response_generation_agent],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

# Extract outputs
task_outputs = [task.output.raw for task in crew.tasks]

# Create DPO entries
entries = [
    {
        "prompt": task_outputs[0],
        "chosen": task_outputs[2],
        "rejected": task_outputs[3],
        "image_name": "image_001.png",
        "type": "descriptive"
    },
    {
        "prompt": task_outputs[1],
        "chosen": task_outputs[2],
        "rejected": task_outputs[3],
        "image_name": "image_001.png",
        "type": "qa"
    }
]

# Save test output
output_file = os.path.join(WORKING_DIR, "dpo_test_output.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

print(f"\n✓ Test output saved to: {output_file}")
print("\nGenerated entries:")
for i, entry in enumerate(entries, 1):
    print(f"\n--- Entry {i} ({entry['type']}) ---")
    print(f"Prompt: {entry['prompt'][:100]}...")
    print(f"Chosen: {entry['chosen'][:100]}...")
    print(f"Rejected: {entry['rejected'][:100]}...")
