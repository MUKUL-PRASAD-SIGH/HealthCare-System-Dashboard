

import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for
import fitz  # PyMuPDF
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded 

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('db.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn


def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_medical_info(text):
    """Extract relevant medical info from text."""
    sections = {
        "Past Diseases": "",
        "Current Symptoms": "",
        "Medications": "",
        "Allergies": "",
        "Surgical History": "",
        "Family Medical History": ""
    }

    # Custom extraction logic (examples)
    sections["Past Diseases"] = "Extracted past diseases text..."
    sections["Current Symptoms"] = "Extracted current symptoms text..."
    sections["Medications"] = "Extracted medications text..."
    sections["Allergies"] = "Extracted allergies text..."
    sections["Surgical History"] = "Extracted surgical history text..."
    sections["Family Medical History"] = "Extracted family medical history text..."

    return sections

# Save extracted sections to CSV function
def save_to_csv(sections, output_path):
    """Save extracted sections to CSV file."""
    df = pd.DataFrame([sections])
    df.to_csv(output_path, index=False)





# PDF Extraction and Medical Info Processing
pdf_path = "UG_First_Year_Syllabus.pdf"  # Update this path with your actual file
past_medical_records_text = extract_text_from_pdf(pdf_path)
extracted_info = extract_medical_info(past_medical_records_text)

# GPT-2 Input
current_symptoms = "fever, cough"
input_text = (
    f"Given the past medical records: {extracted_info['Past Diseases']} "
    f"and current symptoms: {current_symptoms}, "
    "what is the possible disease, recommended medicine, and usage directions?"
)

# Tokenize Input
inputs = tokenizer(input_text, return_tensors='pt')

# Generate Output
outputs = model.generate(
    inputs['input_ids'], 
    max_length=200,  # Set the maximum length for the output
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    temperature=0.7,  # Control randomness
    top_k=50,  # Limit to top-k probable tokens
    top_p=0.95,  # Nucleus sampling (focus on most probable tokens)
)

# Decode and Print Result
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:")
print(result_text)

# Parse Output
parts = result_text.split(',')

def extract_part_value(part, key):
    """Safely extract the value for a given key in the format 'key: value'."""
    if ':' in part:
        k, v = part.split(':', 1)
        if k.strip().lower() == key.lower():
            return v.strip()
    return "Not found"


    disease = extract_part_value(parts[0], "disease") if len(parts) > 0 else "Not found"
    medicine = extract_part_value(parts[1], "medicine") if len(parts) > 1 else "Not found"
    directions = extract_part_value(parts[2], "directions") if len(parts) > 2 else "Not found"

    # Render the results template with extracted information
    return render_template('result.html', disease=disease, medicine=medicine, directions=directions)

    # Render the home page template
    return render_template('home.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Handle user feedback submission
        user_feedback = request.form['feedback']
        conn = get_db_connection()
        conn.execute('INSERT INTO feedback (content) VALUES (?)', (user_feedback,))
        conn.commit()
        conn.close()
        return redirect(url_for('home'))

    # Render the feedback page template
    return render_template('feedback.html')

# Flask app initialization
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Run the Flask app in debug mode
    app.run(debug=True)



print("\nExtracted Information:")
print(f"Disease: {disease}")
print(f"Medicine: {medicine}")
print(f"Directions: {directions}")

