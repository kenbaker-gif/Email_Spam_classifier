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

@app.get("/")
def root():
    return {
        "name": "African SMS Spam Classifier",
        "version": "1.0.0",
        "author": "Ainebyona Abubaker",
        "endpoints": {
            "classify": "POST /classify",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

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

@app.get("/test")
def test():
    messages = [
        "Your MTN Mobile Money account has won 500,000 UGX! Dial *165# now to claim your prize before it expires.",
        "URGENT: Your Airtel Money PIN needs verification. Reply now or your account will be suspended immediately.",
        "Congratulations! You have been selected as an MTN loyalty winner. Send your National ID to claim 1,000,000 UGX.",
        "STANBIC BANK: Your account has been flagged. Click http://stanbic-verify.net to update your details immediately.",
        "Win big! MTN is giving UGX 5,000,000 to 10 lucky customers. Send LUCKY to 8080. Cost: 500 UGX per SMS.",
        "Congratulations! MTN has selected you for a free iPhone 15. Send UGX 5,000 processing fee to 0771234567.",
        "Your NSSF benefits of UGX 4,500,000 are ready. Send your National ID to 0701111222 to process withdrawal.",
        "FREE AIRTIME: Forward this message to 10 contacts and receive UGX 10,000 airtime instantly on your MTN line.",
        "POLICE ALERT: A warrant has been issued in your name. Call immediately to resolve: 0700123456.",
        "You have a pending mobile money transfer of UGX 750,000. Confirm your PIN by replying to this message.",
        "Dear customer, your MoMo account shows suspicious activity. Reply with your PIN to secure your account now.",
        "MTN ALERT: Your SIM will be deactivated in 24 hours. Verify your details at bit.ly/mtn-verify now!",
        "Equity Bank Uganda: Unusual transaction detected. Your card ending 4521 was used. Reply 1 to confirm or 2 to block.",
        "Airtel promotion: Your number has been randomly selected. Dial *170# and enter promo code WIN2024 to get your prize.",
        "MINISTRY OF HEALTH: You qualify for free health insurance. Register by sending your NIN to 0800100200 today.",
        "Your MTN Mobile Money transfer of UGX 50,000 to John Mukasa (0701234567) was successful. Balance: UGX 123,450.",
        "Airtel Money: You have received UGX 30,000 from Sarah Nalwoga. Your new balance is UGX 85,200.",
        "DFCU Bank: Your salary credit of UGX 1,500,000 has been received. Available balance: UGX 1,623,000.",
        "Makerere University: Your tuition payment of UGX 800,000 has been received for Semester 1 2024/2025.",
        "Umeme token for meter 45678123: 4521-8763-1209-3847-6521. Units: 45.2 kWh. Thank you.",
        "Your Jumia order #JM-887234 has been dispatched and will arrive in 2-3 business days.",
        "MTN: Your monthly bundle of 5GB data has been renewed successfully. Valid for 30 days. Dial *156# to check balance.",
        "NSSF Uganda: Your monthly contribution of UGX 45,000 for October has been received. Balance: UGX 2,340,000.",
        "Stanbic Bank: Your transaction of UGX 200,000 at Nakumatt Oasis has been processed successfully.",
        "URA: Your tax return for FY 2023/2024 has been successfully submitted. Reference: URA-2024-78234.",
        "Hi John, the meeting is confirmed for tomorrow at 9am in the boardroom. Please bring the Q3 report.",
        "Your Airtel bundle of 2GB expires tomorrow. Dial *175# to renew and stay connected.",
        "Uganda Airlines: Your booking reference KQ2341 is confirmed. Flight departs Entebbe 14:30 on 15-Nov-2024.",
        "NWSC: Your water bill of UGX 45,000 is due on 30th November. Pay via MTN MoMo paybill 200456.",
        "Your KCB loan repayment of UGX 200,000 for November has been received. Outstanding balance: UGX 800,000.",
    ]

    results = []
    for msg in messages:
        result = classifier(msg)[0]
        results.append({
            "message": msg[:70] + "..." if len(msg) > 70 else msg,
            "label": result["label"],
            "confidence": round(result["score"], 4),
            "is_spam": result["label"] == "SPAM"
        })
    
    spam_count = sum(1 for r in results if r["is_spam"])
    ham_count = len(results) - spam_count

    return {
        "total": len(results),
        "spam_detected": spam_count,
        "ham_detected": ham_count,
        "results": results
    }

@app.post("/classify/batch")
def classify_batch(messages: list[Message]):
    results = []
    for msg in messages:
        result = classifier(msg.text)[0]
        results.append({
            "text": msg.text[:70],
            "label": result["label"],
            "confidence": round(result["score"], 4),
            "is_spam": result["label"] == "SPAM"
        })
    return {"total": len(results), "results": results}