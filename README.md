Spam SMS Detection System 📩🚫

A Machine Learning-based Spam SMS Detection System developed using Python, Natural Language Processing (NLP), and Scikit-learn. The application classifies SMS messages as Spam or Ham (Not Spam) in real time using TF-IDF vectorization and supervised machine learning algorithms.

The project integrates a user-friendly Tkinter graphical interface for interactive message testing and prediction with confidence scoring.

🔍 Project Overview

Spam messages and phishing SMS attacks have become increasingly common in digital communication. This project aims to build an intelligent SMS filtering system capable of identifying spam messages automatically using machine learning and NLP preprocessing techniques.

The system processes raw SMS text, transforms it into numerical feature vectors using TF-IDF, and predicts whether the message is spam using trained classification models.

🚀 Key Features
Real-time SMS spam classification
NLP-based text preprocessing pipeline
TF-IDF feature extraction
Logistic Regression and SVM classifiers
Confidence percentage prediction
Interactive Tkinter GUI
High accuracy spam detection
Modular and scalable project structure
🛠 Technologies Used
Technology	Purpose
Python	Core programming language
Scikit-learn	Machine learning models
Pandas	Dataset handling
NLTK	NLP preprocessing
TF-IDF	Text vectorization
Tkinter	GUI development
Joblib	Model serialization
📂 Project Structure
SpamSMSDetection/
│
├── dataset/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── preprocess.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation and Setup
1. Clone the Repository
git clone https://github.com/your-username/Spam-SMS-Detection.git
2. Navigate to Project Directory
cd Spam-SMS-Detection
3. Create Virtual Environment
python -m venv venv
Activate Virtual Environment
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
4. Install Required Dependencies
pip install -r requirements.txt
▶️ Model Training

Run the following command to preprocess the dataset, train the machine learning models, and save the trained model files:

python train_model.py
▶️ Run the Application

Launch the GUI-based Spam SMS Detection application:

python app.py
🧠 Machine Learning Workflow
SMS Input
   ↓
Text Preprocessing
   ↓
Tokenization & Stopword Removal
   ↓
TF-IDF Vectorization
   ↓
Model Prediction (SVM / Logistic Regression)
   ↓
Spam / Not Spam Classification
📊 NLP Preprocessing Techniques

The following Natural Language Processing techniques are implemented:

Lowercase conversion
Removal of special characters
Tokenization
Stopword removal
Stemming
TF-IDF vectorization
📈 Models Used
Logistic Regression

Used as a baseline supervised classification algorithm for spam prediction.

Support Vector Machine (SVM)

Used for high-performance text classification with improved spam detection accuracy.

🎯 Sample Spam Messages
URGENT! You have won a FREE vacation. Call now!
Congratulations! You won a lottery worth $10000.
🔮 Future Enhancements
Flask-based web deployment
Dark mode interface
SMS API integration
Phishing URL detection
Email spam filtering
Android SMS integration
Deep learning-based classification
