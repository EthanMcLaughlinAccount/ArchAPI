from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load your model from Hugging Face Hub
classifier = pipeline("text-classification", model="neuraxcompany/text-classification-arch")

# Define input request schema
class InputData(BaseModel):
    statement: str
    age: int

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    # Combine statement + age as input (if that's how you trained it)
    full_input = f"Statement: {data.statement} Age: {data.age}"
    
    result = classifier(full_input, return_all_scores=False)
    
    # Extract prediction label
    prediction = result[0]['label']
    return {"archetype": prediction}

