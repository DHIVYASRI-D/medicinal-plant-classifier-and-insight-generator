import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

INSIGHT_PROMPT_TEMPLATE = """
Generate clear and well-structured insights for the medicinal plant "{plant_name}" under the following four sections:

## Medicinal Uses
- List important health benefits or diseases this plant helps with.

## How to Use
- Describe how the plant is typically used (e.g., as tea, crushed leaves, paste).

## Growing Tips
- Give 2–4 simple tips for growing the plant at home or in a garden.

## Precautions
- Mention any warnings, side effects, or who should avoid it.

Respond with only the 4 sections and bullet points. No introduction or conclusion.
"""

def generate_insight_for_plant(plant_name: str) -> dict:
    prompt = INSIGHT_PROMPT_TEMPLATE.format(plant_name=plant_name)

    try:
        chat = model.start_chat()
        response = chat.send_message(prompt)
        raw_output = response.text
        print("=== Raw Gemini Output ===")
        print(raw_output)

        return parse_gemini_response(raw_output)

    except Exception as e:
        print(f"Error generating insights: {e}")
        return {
            "Medicinal Uses": [],
            "How to Use": [],
            "Growing Tips": [],
            "Precautions": [],
        }

def parse_gemini_response(response_text: str) -> dict:
    sections = ["Medicinal Uses", "How to Use", "Growing Tips", "Precautions"]
    insights = {}

    for i, section_title in enumerate(sections):
        next_section_title = sections[i + 1] if i + 1 < len(sections) else None

        pattern = f"## {re.escape(section_title)}(.*?)"
        if next_section_title:
            pattern += f"(?=## {re.escape(next_section_title)})"
        else:
            pattern += r"$"

        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            lines = [line.strip("-• ").strip() for line in content.splitlines() if line.strip()]
            insights[section_title] = lines
        else:
            insights[section_title] = []

    return insights
