import pandas as pd
import re
from flask import Flask, render_template, request, session, redirect, url_for
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
app = Flask(__name__)
app.secret_key = os.urandom(24)
data = pd.read_csv(r"D:\MEDMAX\MEDMAX\o.csv", on_bad_lines='skip')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

data['clean_symptom'] = data['symptom'].apply(clean_text)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

dataset_embeddings = embed_model.encode(data['clean_symptom'].tolist(), convert_to_tensor=True)

def predict_medicine(user_input):
    input_clean = clean_text(user_input)
    input_embedding = embed_model.encode(input_clean, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, dataset_embeddings)
    cosine_scores_np = cosine_scores.cpu().numpy()
    best_idx = np.argmax(cosine_scores_np)
    pred_keyword = data.iloc[best_idx]['keyword']
    
    medicines = data.loc[data['keyword'] == pred_keyword, 'medicine'].unique()[:1]
    medicine = ', '.join(medicines)

    if medicine:
        return f"{medicine}. Please consult a doctor before taking any medication.Consult a doctor if the symptoms persist throughout"
    else:
        return "I am unable to find a specific medicine for your symptoms. Please consult a healthcare professional."

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = [
            {'sender': 'medmax', 'text': 'Hello! I am MedMax, your AI medical assistant. How can I help you today?'}
        ]
    return render_template('checkin.html', chat_history=session['chat_history'])

@app.route('/predict', methods=['POST'])
def predict():
    user_symptom = request.form.get('symptom', '')
    if user_symptom:

        session['chat_history'].append({'sender': 'user', 'text': user_symptom})
        
        medicine_advice = predict_medicine(user_symptom)
        session['chat_history'].append({'sender': 'medmax', 'text': medicine_advice})
        
        session.modified = True
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    if os.path.exists(r'C:\Users\Praveenkumar\OneDrive\Desktop - Copy\Desktop\PROGRAMS\MEDMAX\o.csv'):
        app.run(debug=True)
    else:
        print("Error: 'o.csv' file not found. Please place the CSV file in the same directory as App.py.")