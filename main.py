from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from transformers import pipeline

# --- AI Model Loading ---
# Load the spaCy model for entity recognition
nlp_entities = spacy.load("en_core_web_sm")

# Load a pre-trained pipeline for summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load a pre-trained pipeline for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# --- Data Models ---
# Define the input data shape
class Document(BaseModel):
    text: str


# --- FastAPI Application ---
# Create the FastAPI app instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Document Analyst. Go to /docs to see the API documentation."}


@app.post("/analyze")
def analyze_document(doc: Document):
    # --- 1. Entity Recognition ---
    processed_doc = nlp_entities(doc.text)
    entities = []
    for ent in processed_doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    # --- 2. Summarization ---
    summary_result = summarizer(doc.text, max_length=150, min_length=30, do_sample=False)
    summary = summary_result[0]['summary_text']

    # --- 3. Sentiment Analysis ---
    sentiment_result = sentiment_analyzer(doc.text)
    # The result is a list with one dictionary, so we extract its contents
    sentiment = sentiment_result[0]

    # --- 4. Combine All Results ---
    # Combine all our analyses into a single response object
    analysis_result = {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment
    }

    return analysis_result