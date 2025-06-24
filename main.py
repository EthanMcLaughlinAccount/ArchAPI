from fastapi import FastAPI, Request
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="neuraxcompany/text-classification-arch"
)

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/classify")
async def classify(request: Request):
    data = await request.json()
    text = data.get("text", "")
    result = classifier(text)
    return {"result": result}
