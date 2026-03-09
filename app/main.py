from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load model once at startup directly from HF Hub
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model="kenbaker-gif/African_SMS_Spam_Classifier"
)
print("Model loaded!")

class Message(BaseModel):
    text: str

@app.post("/classify")
def classify(msg: Message):
    try:
        result = classifier(msg.text)[0]
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