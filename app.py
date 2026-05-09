import tkinter as tk
import joblib

from preprocess import clean_text

# load trained model
model = joblib.load("models/spam_model.pkl")

# load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# prediction function
def predict_sms():

    sms = text_box.get("1.0", tk.END)

    cleaned_sms = clean_text(sms)

    vectorized_sms = vectorizer.transform([cleaned_sms])

    prediction = model.predict(vectorized_sms)

    probability = model.predict_proba(vectorized_sms)

    spam_probability = probability[0][1] * 100

    ham_probability = probability[0][0] * 100

    if prediction[0] == 1:

        result_label.config(
            text=f"SPAM MESSAGE\nConfidence: {spam_probability:.2f}%",
            fg="red"
        )

    else:

        result_label.config(
            text=f"NOT SPAM\nConfidence: {ham_probability:.2f}%",
            fg="green"
        )

# create window
window = tk.Tk()

window.title("Spam SMS Detector")

window.geometry("500x400")

# heading
title_label = tk.Label(
    window,
    text="Spam SMS Detection",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=20)

# text box
text_box = tk.Text(
    window,
    height=10,
    width=50,
    font=("Arial", 12)
)

text_box.pack(pady=10)

# button
check_button = tk.Button(
    window,
    text="Check Message",
    command=predict_sms,
    font=("Arial", 12, "bold")
)

check_button.pack(pady=10)

# result
result_label = tk.Label(
    window,
    text="",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=20)

# run app
window.mainloop()