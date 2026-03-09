from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/kenbaker-gif/African_SMS_Spam_Classifier/pipeline/text-classification"

class Message(BaseModel):
    text: str

@app.post("/classify")
def classify(msg: Message):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": msg.text}
        )
        print(f"HF Status: {response.status_code}")
        print(f"HF Response: {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF error {response.status_code}: {response.text}")

        result = response.json()[0][0]
        return {
            "label": result["label"],
            "confidence": round(result["score"], 4),
            "is_spam": result["label"] == "SPAM"
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}