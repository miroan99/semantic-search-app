import os
from dotenv import load_dotenv

# Optional libraries you might use soon
# from openai import OpenAI
# from huggingface_hub import InferenceClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

def health_check():
    print("🚀 Project boot OK")
    print(f"OPENAI_API_KEY set: {'YES' if bool(OPENAI_API_KEY) else 'NO'}")
    print(f"HF_API_KEY set: {'YES' if bool(HF_API_KEY) else 'NO'}")

if __name__ == "__main__":
    health_check()
