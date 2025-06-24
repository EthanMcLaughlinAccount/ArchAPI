from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI()

# Load Hugging Face model pipeline
classifier = pipeline("text-classification", model="neuraxcompany/text-classification-arch")

# Define expected request body
class InputData(BaseModel):
    statement: str
    age: int

@app.get("/")
def root():
    return {"message": "Archetype Classification API is running"}

# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    # Combine statement and age as input
    full_input = f"Statement: {data.statement} Age: {data.age}"
    result = classifier(full_input, return_all_scores=False)
    prediction = result[0]['label']
    score = result[0]['score']
    return {"archetype": prediction, "confidence": score}
