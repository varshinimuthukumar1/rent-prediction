import json

import google.generativeai as genai

from .config import API_KEY, MODEL_NAME, SYSTEM_PROMPT


class GeminiFeatureExtractor:
    def __init__(self):
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env in the project root.")
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(MODEL_NAME)

    def process_batch(self, descriptions):
        full_prompt = f"{SYSTEM_PROMPT}\n\nListings to analyze:\n{json.dumps(descriptions)}"
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
            raw_text = getattr(response, "text", None) or ""
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            if not clean_text:
                print("Error processing batch: model returned empty response (possible block or filter).")
                return [None] * len(descriptions)
            # Try to pull out a JSON array if model wrapped it in prose
            clean_text = self._extract_json_array(clean_text) or clean_text
            data = json.loads(clean_text)
            if isinstance(data, list) and len(data) == len(descriptions):
                return data
            if isinstance(data, list):
                return data + [None] * (len(descriptions) - len(data))
            return [None] * len(descriptions)
        except json.JSONDecodeError as e:
            print(f"Error processing batch (JSON): {e}")
            raw_preview = (raw_text if "raw_text" in dir() else "")[:300]
            if raw_preview:
                print(f"  Response preview: {raw_preview!r}")
            try:
                salvage = self._salvage_json_list(clean_text)
                if salvage:
                    return salvage + [None] * (len(descriptions) - len(salvage))
            except Exception:
                pass
            return [None] * len(descriptions)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return [None] * len(descriptions)

    @staticmethod
    def _extract_json_array(text):
        """If the model wrapped JSON in prose or markdown, try to extract the array."""
        if not text or not isinstance(text, str):
            return None
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("[")
        if start == -1:
            return None
        depth = 1
        i = start + 1
        in_str = False
        escape = False
        q = None
        while i < len(text) and depth:
            c = text[i]
            if escape:
                escape = False
                i += 1
                continue
            if in_str:
                if c == "\\":
                    escape = True
                elif c == q:
                    in_str = False
                i += 1
                continue
            if c in ('"', "'"):
                in_str, q = True, c
                i += 1
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            i += 1
        return text[start:i] if depth == 0 else None

    @staticmethod
    def _salvage_json_list(text):
        """Try to extract a valid list from truncated or malformed JSON."""
        if not text or not isinstance(text, str):
            return []
        text = text.replace("```json", "").replace("```", "").strip()
        if not text.startswith("["):
            return []
        # Try closing the array
        for suffix in ("]", "}]", "}]}"):
            try:
                return json.loads(text + suffix)
            except json.JSONDecodeError:
                pass
        # Try trimming from the end until something parses
        for end in range(len(text), max(0, len(text) - 500), -1):
            try:
                return json.loads(text[:end] + "]")
            except json.JSONDecodeError:
                continue
        return []