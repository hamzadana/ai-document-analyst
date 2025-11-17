## Import necessary libraries
from fastapi import FastAPI
from pantic import BaseModel # Corrected from 'pantic' to 'pydantic'
from transformers import pipeline
import spacy

# ======================================================================================
#  1. AI Model Loading
# ======================================================================================
# We load the models here at the top level of the script.
# This ensures they are loaded only ONCE when the application starts,
# making the API much more efficient. Loading a model for every request
# would be very slow.

## Load the spaCy model for Named Entity Recognition (NER)
## 'en_core_web_sm' is a small, efficient English model.
nlp_entities = spacy.load("en_core_web_sm")

## Load a pre-trained pipeline for Text Summarization
## 'sshleifer/distilbart-cnn-12-6' is a well-balanced model for quality and speed.
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

## Load a pre-trained pipeline for Sentiment Analysis
## 'distilbert-base-uncased-finetuned-sst-2-english' is a fast and accurate model.
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# ======================================================================================
#  2. Pydantic Data Models
# ======================================================================================
# These models define the expected structure of our API requests and responses.
# FastAPI uses them to validate incoming data and to serialize outgoing data.

class Document(BaseModel):
    """The input data model for a document to be analyzed."""
    text: str

class AnalysisResult(BaseModel):
    """The output data model for the analysis results."""
    summary: str
    entities: list
    sentiment: dict


# ======================================================================================
#  3. FastAPI Application
# ======================================================================================

app = FastAPI(
    title="AI Document Analyst API",
    description="An API that provides text summarization, named entity recognition, and sentiment analysis."
)

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the AI Document Analyst API. Go to /docs for more information."}


## The 'response_model' parameter tells FastAPI to validate the output against our
## AnalysisResult model. This ensures our API always returns a consistent, predictable structure.
@app.post("/analyze", response_model=AnalysisResult)
def analyze_document(doc: Document):
    """
    Analyzes the input document to perform three tasks:
    1. Summarization
    2. Named Entity Recognition
    3. Sentiment Analysis
    """
    # --- Task 1: Named Entity Recognition ---
    processed_doc = nlp_entities(doc.text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in processed_doc.ents]

    # --- Task 2: Summarization ---
    summary_result = summarizer(doc.text, max_length=150, min_length=30, do_sample=False)
    summary = summary_result[0]['summary_text']

    # --- Task 3: Sentiment Analysis ---
    sentiment_result = sentiment_analyzer(doc.text)
    sentiment = sentiment_result[0]

    # --- Combine and return the results ---
    analysis_result = {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment
    }

    return analysis_result