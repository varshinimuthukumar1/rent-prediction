import google.generativeai as genai
import json
from .config import API_KEY, MODEL_NAME, SYSTEM_PROMPT

class GeminiFeatureExtractor:
    def __init__(self):
        if not API_KEY:
            raise ValueError("API Key not found. Check your .env file.")
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(MODEL_NAME)

    def process_batch(self, descriptions):
        full_prompt = f"{SYSTEM_PROMPT}\n\nListings to analyze:\n{json.dumps(descriptions)}"
        
        try:
            response = self.model.generate_content(full_prompt)
            # Removes markdown backticks if Gemini includes them
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return [None] * len(descriptions)