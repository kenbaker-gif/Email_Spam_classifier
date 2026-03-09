from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

class Message(BaseModel):
    text: str

@app.post("/classify")
def classify(msg: Message):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    result = client.text_classification(
        msg.text,
        model="kenbaker-gif/African_SMS_Spam_Classifier"
    )
    
    top = result[0]
    return {
        "label": top.label,
        "confidence": round(top.score, 4),
        "is_spam": top.label == "SPAM"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

And update `requirements.txt`:
```
fastapi
uvicorn
huggingface-hub
python-dotenv
pydantic