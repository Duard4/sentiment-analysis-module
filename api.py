import pickle
import pymorphy3
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Load the model and vectorizer
try:
    model = joblib.load("models/log_reg.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
except FileNotFoundError:
    print("Error: Model files not found. Make sure the models directory exists with the necessary files.")
    exit(1)

import re
from pymorphy3 import MorphAnalyzer
import stopwordsiso as stopwords

# Initialize components once (could be moved outside the function for better performance)
stopwords_uk = stopwords.stopwords("uk")
morph = MorphAnalyzer(lang='uk')

NEGATIONS = {"не", "ні", "жоден"}
STANDALONE_NEGATIONS = {"ніхто", "ніщо", "ніде"} 

def preprocess(text):
    """Preprocess a single text for sentiment analysis using the full pipeline."""
    # Initial text cleaning
    text = re.sub(r'^Review\s*', '', text)  # Remove leading "Review" if present
    text = text.strip()
    text = re.sub(r'[^\w\s]', ' ', text)    # Replace punctuation with spaces
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
    
    # Tokenization
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 1]  # Remove single-character tokens
    
    # Negation handling
    def merge_negations(tokens):
        merged = []
        i = 0
        while i < len(tokens):
            current = tokens[i].lower()
            
            if current in STANDALONE_NEGATIONS:
                merged.append(current)
                i += 1
                
            elif current in NEGATIONS and i + 1 < len(tokens):
                next_word = tokens[i+1].lower()
                if next_word not in STANDALONE_NEGATIONS:  
                    merged.append(f"{current}_{next_word}")
                    i += 2
                else:
                    merged.append(current)
                    i += 1
            else:
                merged.append(current)
                i += 1
        return merged
    
    tokens = merge_negations(tokens)
    
    # Lemmatization
    def lemmatize_uk(token):
        if "_" in token:
            neg, word = token.split("_", 1)
            lemma = morph.parse(word)[0].normal_form
            return f"{neg}_{lemma}"
        return morph.parse(token)[0].normal_form
    
    lemmas = [lemmatize_uk(token) for token in tokens]
    
    # Stopword removal (keeping negation constructs)
    filtered = [
        lemma for lemma in lemmas
        if ("_" in lemma) or (lemma not in stopwords_uk)
    ]
    
    # Join back to text for vectorization
    processed_text = ' '.join(filtered)
    
    return processed_text

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze sentiment of text provided in request."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
    
    try:
        # Preprocess and predict
        text_vectorized = vectorizer.transform([preprocess(text)])
        sentiment_id = model.predict(text_vectorized)[0]
        sentiment_map = {0: "Негативний", 1: "Нейтральний", 2: "Позитивний"}
        sentiment = sentiment_map[sentiment_id]
        
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "sentiment_id": int(sentiment_id)
        })
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Ukrainian Sentiment Analysis"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)