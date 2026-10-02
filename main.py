# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sys

app = FastAPI(title="Domain Categorization API")

def skill_tokenizer(text):
    return [t.strip().lower() for t in text.split(',') if t.strip()]

import __main__
__main__.skill_tokenizer = skill_tokenizer

vec = joblib.load("domain_tfidf_v1.pkl")
clf = joblib.load("domain_clf_v1.pkl")

VALID_DOMAINS = set(clf.classes_)


class CategorizeRequest(BaseModel):
    skills_text: str


@app.get("/")
def home():
    return {"message": "Domain Categorization API is running"}


@app.get("/domains")
def domains():
    return {"domains": sorted(VALID_DOMAINS)}


@app.post("/categorize")
def categorize(req: CategorizeRequest):
    if not req.skills_text.strip():
        raise HTTPException(400, "skills_text cannot be empty")

    x = vec.transform([req.skills_text])
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0].max()

    return {
        "predicted_domain": str(pred),
        "confidence": round(float(proba), 3)
    }