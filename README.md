# AI Document Analyst API

## Overview

This project is a comprehensive AI-powered API built with FastAPI. It serves as a backend service that can analyze a block of text and return a structured JSON object containing three key insights: a concise summary, a list of extracted named entities (like people, places, and organizations), and a sentiment analysis score (positive or negative).

This project was built from the ground up as a capstone portfolio piece to demonstrate a full development lifecycle, from local environment setup to a functional AI application.

## Features

- **Named Entity Recognition (NER):** Identifies and extracts real-world entities from the text.
- **Text Summarization:** Generates a short, concise summary of the provided document.
- **Sentiment Analysis:** Determines the underlying sentiment of the text as either POSITIVE or NEGATIVE with a confidence score.
- **Robust API:** Built with FastAPI, providing automatic interactive documentation (via Swagger UI).

## Technologies Used

- **Backend:** Python, FastAPI
- **Natural Language Processing (NLP):**
  - **spaCy:** For fast and efficient Named Entity Recognition.
  - **Hugging Face Transformers:** For state-of-the-art Summarization and Sentiment Analysis.
- **Server:** Uvicorn
- **Environment Management:** venv, pip

## Local Installation & Usage

To run this project on your local machine, please follow these steps.

**1. Clone the Repository:**
```bash
git clone https://github.com/hamzadana/ai-document-analyst.git
cd ai-document-analyst