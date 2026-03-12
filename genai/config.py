import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of genai/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemma-3-27b-it" 

# The "Expert" Prompt
SYSTEM_PROMPT = """
Act as a German Real Estate expert. Analyze the provided apartment descriptions.
Extract features ONLY if they are explicitly mentioned in the text.
If a feature is not mentioned, return null.

Do NOT infer or guess.

Extract the following features into a JSON list:
- luxury_score: (Integer: range(1-10))
- floor_heating: (Boolean)
- guest_toilet: (Boolean)
- built_in_kitchen: (Boolean)
- garage_available: (Boolean)
- air_conditioning: (Boolean)
- dishwasher: (Boolean)
- bathtub: (Boolean)
- laminate_floor: (Boolean)
- parquet_floor: (Boolean)
- double_glazing: (Boolean)
- green_view: (Boolean)
- quiet_neighborhood: (Boolean)
- near_public_transport: (Boolean)
"""