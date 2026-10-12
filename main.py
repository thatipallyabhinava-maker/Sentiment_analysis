
from flask import Flask, render_template, request
from textblob import TextBlob
from transformers import pipeline
from functools import lru_cache

app = Flask(__name__)

print("Loading sentiment analysis model...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

print("Model loaded! Start analyzing text.")


@lru_cache(maxsize=256)
def analyze_text(text):
    # Get prediction from the pretrained AI model
    result = sentiment_model(
        text,
        truncation=True,
        max_length=512
    )[0]

    sentiment = result["label"].lower()
    confidence = round(result["score"] * 100, 2)

    # Calculate polarity and subjectivity
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    return sentiment.title(), confidence, polarity, subjectivity


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    confidence = 0
    polarity = 0
    subjectivity = 0
    user_text = ""
    error = ""

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()

        if not user_text:
            error = "Please enter some text."

        elif len(user_text) > 5000:
            error = "Please enter text with fewer than 5000 characters."

        else:
            try:
                (
                    sentiment,
                    confidence,
                    polarity,
                    subjectivity
                ) = analyze_text(user_text)

            except Exception as e:
                app.logger.exception("Sentiment analysis failed")
                error = "Analysis failed. Please try again."

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        polarity=polarity,
        subjectivity=subjectivity,
        user_text=user_text,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
