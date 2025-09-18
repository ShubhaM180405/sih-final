import pandas as pd
import re
from transformers import pipeline
from utils.utils import match_keywords

# Load model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# Load keywords from CSV
def load_keywords(path: str) -> list:
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {path} with available encodings")
        if 'keyword' not in df.columns:
            raise ValueError(f"{path} must contain a 'keyword' column")
        keywords = df['keyword'].dropna().astype(str).str.lower().str.strip().tolist()
        if not keywords:
            raise ValueError(f"{path} contains no valid keywords")
        return keywords
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return []

NEGATIVE_KEYWORDS = load_keywords("keywords_negative.csv")
POSITIVE_KEYWORDS = load_keywords("keywords_positive.csv")
NEUTRAL_KEYWORDS = load_keywords("keywords_neutral.csv")

def adjust_sentiment(text: str, sentiment: str, score: float) -> tuple:
    """
    Refine sentiment using keyword counts + model confidence.
    Returns (main_sentiment, sub_sentiment).
    """
    if not text.strip():
        return "Neutral", "Neutral (Pure Neutral)"
    text_lower = text.lower()
    pos_hits = match_keywords(text_lower, POSITIVE_KEYWORDS)
    neg_hits = match_keywords(text_lower, NEGATIVE_KEYWORDS)
    neu_hits = match_keywords(text_lower, NEUTRAL_KEYWORDS)
    
    # If neutral keywords are present and sentiment is not strongly confident, consider Neutral
    if neu_hits > 0 and score < 0.9:
        main_sentiment = "Neutral"
        if pos_hits > neg_hits:
            sub_sentiment = "Neutral (Dominantly Positive)"
        elif neg_hits > pos_hits:
            sub_sentiment = "Neutral (Dominantly Negative)"
        else:
            sub_sentiment = "Neutral (Pure Neutral)"
    else:
        main_sentiment = sentiment
        sub_sentiment = sentiment
        if sentiment == "Neutral":
            if neg_hits > pos_hits:
                sub_sentiment = "Neutral (Dominantly Negative)"
            elif pos_hits > neg_hits:
                sub_sentiment = "Neutral (Dominantly Positive)"
            else:
                sub_sentiment = "Neutral (Pure Neutral)"
        elif sentiment == "Positive" and score < 0.9 and neg_hits >= pos_hits + 2:
            sub_sentiment = "Neutral (Dominantly Negative)"
        elif sentiment == "Negative" and score < 0.9 and pos_hits >= neg_hits + 2:
            sub_sentiment = "Neutral (Dominantly Positive)"
    
    return main_sentiment, sub_sentiment

def analyze_sentiment(text: str) -> dict:
    """Analyze single comment with refined Neutral handling."""
    if not text.strip():
        return {
            "text": text,
            "sentiment_main": "Neutral",
            "sentiment_sub": "Neutral (Pure Neutral)",
            "score": 0.0
        }
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    if label == "LABEL_0":
        sentiment = "Negative"
    elif label == "LABEL_1":
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    main_sentiment, sub_sentiment = adjust_sentiment(text, sentiment, result["score"])
    return {
        "text": text,
        "sentiment_main": main_sentiment,
        "sentiment_sub": sub_sentiment,
        "score": round(result["score"], 3)
    }

def analyze_batch(comments: list) -> list:
    """Analyze batch of comments with refined Neutral handling."""
    results = []
    for comment in comments:
        if isinstance(comment, dict) and "text" in comment:
            text = comment["text"]
        else:
            text = str(comment)
        result = analyze_sentiment(text)
        result.update({k: v for k, v in comment.items() if k != "text"})
        results.append(result)
    return results
