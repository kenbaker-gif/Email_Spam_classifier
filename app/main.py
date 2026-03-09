from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/kenbaker-gif/African_SMS_Spam_Classifier"

class Message(BaseModel):
    text: str

@app.post("/classify")
def classify(msg: Message):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
                "x-use-cache": "0"
            },
            json={"inputs": msg.text}
        )

        print(f"HF Status: {response.status_code}")
        print(f"HF Response: {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF error {response.status_code}: {response.text}")

        data = response.json()
        # Response is [[{label, score}, {label, score}]]
        top = max(data[0], key=lambda x: x["score"])
        return {
            "label": top["label"],
            "confidence": round(top["score"], 4),
            "is_spam": top["label"] == "SPAM"
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}