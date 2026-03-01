from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Comment(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(data: Comment):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": data.comment}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return response.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
