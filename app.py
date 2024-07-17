!pip install pymupdf
import fitz  # PyMuPDF
import os
import nltk
# ... rest of your code ...
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
import numpy as np

# Step 1: Parsing Resumes and Job Descriptions
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

# Step 2: Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Step 3: Feature Extraction
def extract_features(resumes, job_description):
    vectorizer = TfidfVectorizer()
    all_texts = resumes + [job_description]
    features = vectorizer.fit_transform(all_texts)
    return features[:-1], features[-1]

# Step 4: Implementing KNN and Calculating Similarity Scores
def find_best_matches(resume_features, job_description_features, k=3):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(resume_features)
    distances, indices = knn.kneighbors([job_description_features])
    similarity_scores = 1 - distances[0]  # Cosine similarity is 1 - distance
    return indices[0], similarity_scores

# Step 5: Application Interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    resume_files = request.files.getlist('resumes')
    job_description_file = request.files['job_description']
    
    resumes_texts = [extract_text(resume) for resume in resume_files]
    job_description_text = extract_text(job_description_file)
    
    preprocessed_resumes = [preprocess_text(text) for text in resumes_texts]
    preprocessed_job_description = preprocess_text(job_description_text)
    
    resume_features, job_description_features = extract_features(preprocessed_resumes, preprocessed_job_description)
    
    indices, similarity_scores = find_best_matches(resume_features, job_description_features, k=len(resume_files))
    
    results = sorted(zip(indices, similarity_scores), key=lambda x: x[1], reverse=True)
    return render_template('results.html', results=results, resumes=resume_files)

if __name__ == '__main__':
    app.run(debug=True)
