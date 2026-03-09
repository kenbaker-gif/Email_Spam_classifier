# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("text-classification", model="kenbaker-gif/African_SMS_Spam_Classifier")

class Message(BaseModel):
    text: str

@app.post("/classify")
def classify(msg: Message):
    result = classifier(msg.text)[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4),
        "is_spam": result["label"] == "SPAM"
    }

@app.get("/health")
def health():
    return {"status": "ok"}