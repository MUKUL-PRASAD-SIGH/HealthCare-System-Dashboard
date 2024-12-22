import os
from flask import Flask, render_template, redirect, url_for, request, flash
import sqlite3

from jsonschema import ValidationError
import sqlalchemy
import fitz  # PyMuPDF
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from pdf_extractor import *

from flask import Flask, render_template, request
from flask_login import login_required
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired


from forms import RegisterForm, LoginForm
from models import db, User

# Create the form class (This is going to be used for the upload.)
class MedicalForm(FlaskForm):
    past_medical_records = FileField('Past Medical Records', 
        validators=[
            FileRequired(),
            FileAllowed(['pdf'], 'PDF files only!')
        ])
    current_symptoms = TextAreaField('Current Symptoms', 
        validators=[DataRequired()])
    submit = SubmitField('Submit')


# Get the absolute path for the uploads folder
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'thisismysecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# This would create database tables
with app.app_context():
    db.create_all()

# Load trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    try:
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password', 'danger')
    except Exception as e:
        flash('An error occurred during login. Please try again.', 'danger')
        app.logger.error(f'Login error: {str(e)}')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        try:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password=hashed_password
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration error: {str(e)}', 'danger')
            app.logger.error(f'Registration error: {str(e)}')
    else:
        # Print form errors for debugging
        if form.errors:
            print("Form errors:", form.errors)
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{field}: {error}", 'danger')
    
    return render_template('register.html', form=form)


@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    form = MedicalForm()
    
    if form.validate_on_submit():
        try:
            # Create uploads directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            # Get the file and secure the filename
            past_medical_records = form.past_medical_records.data
            from werkzeug.utils import secure_filename
            filename = secure_filename(past_medical_records.filename)
            
            # Create full path
            past_medical_records_path = os.path.join(
                os.path.abspath(app.config['UPLOAD_FOLDER']), 
                filename
            )
            
            # Save the file
            past_medical_records.save(past_medical_records_path)
            
            # Debug print
            print(f"File saved to: {past_medical_records_path}")
            
            current_symptoms = form.current_symptoms.data
            print(f"Current symptoms: {current_symptoms}")
            
            past_medical_records_text = extract_text_from_pdf(past_medical_records_path)
            print(f"Extracted text: {past_medical_records_text[:200]}...")  # Print first 200 chars
            
            extracted_info = extract_medical_info(past_medical_records_text)
            print(f"Extracted info: {extracted_info}")
            
            input_text = f"Given the past medical records: {extracted_info['Past Diseases']} and current symptoms: {current_symptoms}, provide a diagnosis in this exact format: 'Disease: [disease name], Medicine: [medicine name], Directions: [usage directions]'"
            
            print(f"Input to model: {input_text}")
            
            inputs = tokenizer(input_text, return_tensors='pt')
            
            # Add some parameters to control the generation
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model output: {result_text}")
            
            # More robust parsing of the output
            try:
                if 'Disease:' in result_text and 'Medicine:' in result_text and 'Directions:' in result_text:
                    disease = result_text.split('Disease:')[1].split('Medicine:')[0].strip()
                    medicine = result_text.split('Medicine:')[1].split('Directions:')[0].strip()
                    directions = result_text.split('Directions:')[1].strip()
                else:
                    disease = "Unable to determine"
                    medicine = "Please consult a doctor"
                    directions = "Visit your healthcare provider for proper diagnosis"
            except Exception as e:
                print(f"Error parsing model output: {e}")
                disease = "Error in processing"
                medicine = "Please consult a doctor"
                directions = "Visit your healthcare provider for proper diagnosis"
            
            # Clean up the uploaded file after processing
            os.remove(past_medical_records_path)
            
            flash('Analysis complete!', 'success')
            return render_template('result.html', 
                                disease=disease, 
                                medicine=medicine, 
                                directions=directions)
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return render_template('home.html', form=form)
    
    return render_template('home.html', form=form)

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        user_feedback = request.form['feedback']
        conn = get_db_connection()
        conn.execute('INSERT INTO feedback (content) VALUES (?)', (user_feedback,))
        conn.commit()
        conn.close()
        return redirect(url_for('home'))
    return render_template('feedback.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
