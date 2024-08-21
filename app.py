from dotenv import load_dotenv
load_dotenv()

from flask import jsonify, Flask, request
import logging
import base64
import os
import io
# from PIL import Image 
# import pdf2image
import google.generativeai as genai
import pandas as pd
import docx2txt
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
from llama_index.core import SimpleDirectoryReader

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Set up Google Generative AI with multiple API keys
api_list = [
    "AIzaSyA51WTz0t69sBFs8D2ZmLLypKs6X9rIcEI",
    "AIzaSyDlCk6V9XXwHEYJSjSC4-g28N69UgNcVYA"
]
api_key = random.choice(api_list)
print(f"Using API Key: {api_key}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

# Preprocessing function
def preprocessing(document):
    text = document.replace('\n', ' ').replace('\t', ' ').lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token and token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to generate a response using Gemini model
def get_gemini_response(input_text, pdf_content, prompt):
    try:
        response = model.generate_content([input_text, pdf_content, prompt], generation_config=genai.GenerationConfig(
            temperature=0.4
        ))
        return response.text
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        return None

# Function to handle PDF file setup and processing
def input_pdf_setup(uploaded_file):
    try:
        if uploaded_file:
            data = SimpleDirectoryReader(input_files=[uploaded_file]).load_data()
            document_resume = " ".join([doc.text.replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ') for doc in data])
            final = preprocessing(document_resume)
            return final
        else:
            raise FileNotFoundError("No file uploaded")
    except Exception as e:
        logging.error(f"Error processing PDF file: {e}")
        return None

@app.route('/score_resumes', methods=['POST'])
def scoring():
    JD_folder_path = 'temp3/'
    resume_folder_path = 'temp4/'
    score = {}

    try:
        if 'jd_file' not in request.files or 'resumes' not in request.files:
            return jsonify({'error': 'Please provide a JD file and at least one resume file.'}), 400

        jd_file = request.files['jd_file']
        resumes = request.files.getlist('resumes')

        os.makedirs(JD_folder_path, exist_ok=True)
        os.makedirs(resume_folder_path, exist_ok=True)

        jd_file_path = os.path.join(JD_folder_path, jd_file.filename)
        jd_file.save(jd_file_path)
        
        resume_paths = []
        for resume in resumes:
            resume_path = os.path.join(resume_folder_path, resume.filename)
            resume.save(resume_path)
            resume_paths.append(resume_path)

        input_text = input_pdf_setup(jd_file_path)
        if input_text is None:
            return jsonify({'error': 'Error processing the JD file.'}), 500

        for resume_path in resume_paths:
            pdf_content = input_pdf_setup(resume_path)
            if pdf_content is None:
                score[os.path.basename(resume_path)] = 'Error processing resume file.'
                continue

            input_prompt4 = """your are an skilled Topic modelling model. get me the job title mentioned in the job description.
            the output should be in this way, position,industry,sub-industry  and don't give the explaination follow the example provided. e.g. AI-Engineer, IT sector, Product based."""

            response1 = model.generate_content([input_text, input_prompt4], generation_config=genai.GenerationConfig(
                temperature=0.3))
            
            job_title = response1.text if response1 else 'Unknown'
            
            input_prompt3 = f"""
            You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of {job_title} and ATS functionality, 
            your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
            the job description. Don't give the explain or suggestion, just matching percentage.
            """

            response = get_gemini_response(input_text, pdf_content, input_prompt3)
            score[os.path.basename(resume_path)] = str(response).replace('\n','')

        return jsonify(score), 200

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
