from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="African SMS Spam Classifier",
    description="Detects spam in African mobile money SMS messages",
    version="1.0.0"
)

# Load model once at startup
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model="kenbaker-gif/African_SMS_Spam_Classifier"
)
print("Model loaded!")


# ── Models ───────────────────────────────────────────────────

class Message(BaseModel):
    text: str

class BatchRequest(BaseModel):
    messages: list[str]


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "African SMS Spam Classifier",
        "version": "1.0.0",
        "author": "Ainebyona Abubaker",
        "endpoints": {
            "classify": "POST /classify",
            "classify_batch": "POST /classify/batch",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/classify/batch")
def classify_batch(req: BatchRequest):
    try:
        results = []
        for text in req.messages:
            result = classifier(text)[0]
            results.append({
                "message": text[:70] + "..." if len(text) > 70 else text,
                "label": result["label"],
                "confidence": round(result["score"], 4),
                "is_spam": result["label"] == "SPAM"
            })

        spam_count = sum(1 for r in results if r["is_spam"])

        return {
            "total": len(results),
            "spam_detected": spam_count,
            "ham_detected": len(results) - spam_count,
            "results": results
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))