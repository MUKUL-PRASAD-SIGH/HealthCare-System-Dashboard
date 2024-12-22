from app import app, db
from flask import session, request, redirect, render_template
import os
from models import MedicalRecord

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            record = MedicalRecord(user_id=session['user_id'], filename=file.filename, symptoms=symptoms)
            db.session.add(record)
            db.session.commit()
            return "File uploaded successfully!"
    return render_template('dashboard.html')
