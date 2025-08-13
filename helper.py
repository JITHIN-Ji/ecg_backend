import os
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load variables from your .env file

# API keys from .env
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]

def analyze_ecg_classification(uploaded_file_bytes, file_extension="pdf"):
    """
    Analyzes an ECG file using the Gemini API, with fallback for multiple API keys.
    Always returns 'Classification Unavailable' if analysis fails.
    """
    prompt = """
    Analyze the provided ECG report. Based on your analysis, classify the ECG into ONLY ONE of the following four categories:

    1. Normal: All parameters are within normal limits.
    2. MI: Clear evidence of an acute Myocardial Infarction (e.g., ST-segment elevation).
    3. History of MI: Evidence of a past/old/evolved infarct (e.g., pathological Q-waves without acute ST changes).
    4. Abnormal heartbeat: An arrhythmia or conduction block is the primary finding (e.g., Atrial Fibrillation, AV Block, Tachycardia).

    Your entire response must be ONLY the name of one of the four categories listed above. Do not add any explanation, punctuation, or other text.
    """

    for api_key in GEMINI_API_KEYS:
        if not api_key:  # Skip empty keys
            continue

        try:
            genai.configure(api_key=api_key)

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file_bytes)
                tmp_file_path = tmp_file.name

            uploaded_file = genai.upload_file(path=tmp_file_path, display_name="ECG Report")

            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                generation_config={"temperature": 0}
            )

            response = model.generate_content([prompt, uploaded_file])

            # If blocked or empty response
            if not response.parts:
                print(f"[WARNING] Gemini blocked the response. Reason: {response.candidates[0].finish_reason.name}")
                genai.delete_file(uploaded_file.name)
                os.unlink(tmp_file_path)
                continue  # Try next API key

            # Success
            genai.delete_file(uploaded_file.name)
            os.unlink(tmp_file_path)
            return response.text.strip()

        except Exception as e:
            print(f"[ERROR] Gemini classification failed: {e}")
            if 'uploaded_file' in locals():
                try:
                    genai.delete_file(uploaded_file.name)
                except:
                    pass
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            continue  # Try next API key

    # If all API keys fail
    return "Classification Unavailable"
