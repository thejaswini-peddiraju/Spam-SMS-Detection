from flask import Flask, render_template, request
import joblib
import re

from preprocess import clean_text

app = Flask(__name__)

# load trained model
model = joblib.load("models/spam_model.pkl")

# load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""
    warning_text = ""
    sms = ""

    if request.method == "POST":

        sms = request.form["message"]

        # suspicious URL detection
        suspicious_patterns = [
            "http",
            "www",
            ".com",
            "bit.ly",
            ".xyz"
        ]

        for pattern in suspicious_patterns:

            if pattern in sms.lower():

                warning_text = (
                    "⚠ Suspicious Link Detected"
                )

                break

        cleaned_sms = clean_text(sms)

        vectorized_sms = vectorizer.transform([cleaned_sms])

        prediction = model.predict(vectorized_sms)

        probability = model.predict_proba(vectorized_sms)

        spam_probability = probability[0][1] * 100
        ham_probability = probability[0][0] * 100

        if prediction[0] == 1:

            prediction_text = (
                f"SPAM MESSAGE "
                f"(Confidence: {spam_probability:.2f}%)"
            )

        else:

            prediction_text = (
                f"NOT SPAM "
                f"(Confidence: {ham_probability:.2f}%)"
            )

    return render_template(
        "index.html",
        prediction=prediction_text,
        warning=warning_text,
        message=sms
    )


if __name__ == "__main__":
    app.run(debug=True)