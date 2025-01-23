from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify,current_app
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from werkzeug.utils import secure_filename
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
from collections import defaultdict
import json
from flask_mysqldb import MySQLdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import re
import random
import ssl
from groq import Groq
import traceback
from datetime import datetime, timedelta

app = Flask(__name__)

# Logging configuration
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'healthcare_buddy'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Secret key for session
app.secret_key = 'your_secret_key'

mysql = MySQL(app)

# Groq API configuration
groq_client = Groq(api_key="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")

# Load medical knowledge base
with open('symptom_questions.json', 'r') as f:
    SYMPTOMS_QUESTIONS = json.load(f)

with open('diagnoses.json', 'r') as f:
    DIAGNOSES = json.load(f)

with open('medical_knowledge.json', 'r') as f:
    medical_knowledge = json.load(f)

# Load diet plans
with open('diet_plans.json', 'r') as f:
    DIET_PLANS = json.load(f)

def get_description(item, category):
    item_lower = item.lower()
    if category in medical_knowledge and item_lower in medical_knowledge[category]:
        return medical_knowledge[category][item_lower]['description']
    return "No additional information available"

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Download NLTK data
def download_nltk_data():
    try:
        # Disable SSL verification (use with caution)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

# Call the function to download NLTK data
download_nltk_data()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_info(text):
    date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', text)
    date = date_match.group(1) if date_match else "Date not found"

    doctor_match = re.search(r'Dr\.\s+([A-Z][a-z]+ [A-Z][a-z]+)', text)
    doctor = doctor_match.group(1) if doctor_match else "Doctor's name not found"

    findings = []
    lines = text.split('\n')
    capture = False
    for line in lines:
        if re.match(r'findings|observations|results|impression', line.strip().lower()):
            capture = True
            continue
        if capture and line.strip():
            findings.append(line.strip())
        if capture and not line.strip():
            break
    
    findings = findings[:3]
    
    return date, doctor, findings

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
    return text

def summarize_text(text, num_sentences=3):
    try:
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    except LookupError as e:
        print(f"NLTK data not found: {str(e)}")
        return "Unable to generate summary due to missing NLTK data."
    except Exception as e:
        print(f"Error in summarize_text: {str(e)}")
        return "An error occurred while generating the summary."
    
    word_frequencies = defaultdict(int)
    for word in words:
        word_frequencies[word] += 1
    
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]
    
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    summary = " ".join(summary_sentences)
    
    return summary

# Initialize and train the machine learning model
model = None

def initialize_model(diagnoses):
    labels = [d['condition'] for d in diagnoses if 'condition' in d]
    symptoms = [' '.join(d['symptoms']) for d in diagnoses if 'symptoms' in d]
    if not labels or not symptoms:
        raise ValueError("Invalid diagnoses data: missing 'condition' or 'symptoms'")
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(symptoms, labels)
    return model

try:
    model = initialize_model(DIAGNOSES)
except Exception as e:
    print(f"Error initializing model: {str(e)}")

def get_user_medical_history(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT name, description FROM medications WHERE user_id = %s", [user_id])
    medications = cur.fetchall()
    
    cur.execute("SELECT name, description FROM allergies WHERE user_id = %s", [user_id])
    allergies = cur.fetchall()
    
    cur.execute("SELECT name, description FROM illnesses WHERE user_id = %s", [user_id])
    illnesses = cur.fetchall()
    
    cur.close()
    
    return {
        'medications': medications,
        'allergies': allergies,
        'illnesses': illnesses
    }

def check_symptom_relation(symptom, medical_history):
    related_conditions = []
    
    for illness in medical_history['illnesses']:
        if symptom.lower() in illness['description'].lower():
            related_conditions.append(f"Symptom may be related to your existing condition: {illness['name']}")
    
    for medication in medical_history['medications']:
        if symptom.lower() in medication['description'].lower():
            related_conditions.append(f"Symptom may be a side effect of your medication: {medication['name']}")
    
    return related_conditions

def process_medical_info(items, category):
    processed_items = []
    for item in items:
        item = item.strip().lower()
        if item in medical_knowledge[category]:
            processed_items.append({
                'name': medical_knowledge[category][item]['name'],
                'description': medical_knowledge[category][item]['description']
            })
        else:
            processed_items.append({
                'name': item.capitalize(),
                'description': 'No additional information available'
            })
    return processed_items

# Subscription check decorator
def check_subscription_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        cur.execute("SELECT subscription_status, subscription_expiry FROM users WHERE id = %s", [user_id])
        user = cur.fetchone()
        cur.close()
        
        if not user or user['subscription_status'] != 'subscribed' or user['subscription_expiry'] <= datetime.now().date():
            flash('This feature requires an active subscription.', 'warning')
            return redirect(url_for('home'))
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT subscription_status, subscription_expiry FROM users WHERE id = %s", [user_id])
    user = cur.fetchone()
    cur.close()
    
    is_subscribed = user and user['subscription_status'] == 'subscribed' and user['subscription_expiry'] > datetime.now().date()
    
    return render_template('home.html', is_subscribed=is_subscribed)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        
        if user and check_password_hash(user['password'], password):
            # Update the `last_login` field
            cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", [user['id']])
            mysql.connection.commit()
            cur.close()

            session['user_id'] = user['id']
            return jsonify({'success': True}), 200
        else:
            cur.close()
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        emergency_contact_name = request.form['emergencyContactName']
        emergency_contact_phone = request.form['emergencyContactPhone']
        city = request.form['city']
        district = request.form['district']
        state = request.form['state']
        height = request.form.get('height')
        weight = request.form.get('weight')
        blood_type = request.form.get('bloodType')
        # smoking_status = request.form.get('smokingStatus')
        # exercise_habits = request.form.get('exerciseHabits')
        # dietary_restrictions = request.form.get('dietaryRestrictions')
        # occupation = request.form.get('occupation')
        #preferred_language = request.form.get('preferredLanguage')
        primary_care_physician = request.form.get('primaryCarePhysician')
        terms_agreed = 'terms' in request.form
        telemedicine_consent = 'telemedicine' in request.form

        # Validate email and phone
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        phone_regex = r'^\+?[1-9]\d{1,14}$'
        
        if not re.match(email_regex, email):
            flash('Invalid email format', 'error')
            return redirect(url_for('signup'))
        
        if not re.match(phone_regex, phone):
            flash('Invalid phone number format', 'error')
            return redirect(url_for('signup'))

        # Check if email already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cur.fetchone()
        if user:
            cur.close()
            flash('Email already exists. Please use a different email.', 'error')
            return redirect(url_for('signup'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        try:
            # Insert user information
            cur.execute("""
                INSERT INTO users (name, age, gender, number, email, password, 
                emergency_contact_name, emergency_contact_phone, 
                city, district, state, height, weight, blood_type, terms_agreed, telemedicine_consent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, age, gender, phone, email, hashed_password, 
                  emergency_contact_name, emergency_contact_phone, 
                  city, district, state, height, weight, blood_type,  
                terms_agreed,telemedicine_consent))
            mysql.connection.commit()

            # Retrieve user ID
            user_id = cur.lastrowid

            # Process medications, allergies, and illnesses
            medications = request.form.get('medications', '').split(',')
            allergies = request.form.get('allergies', '').split(',')
            illnesses = request.form.get('illnesses', '').split(',')

            for medication in medications:
                if medication.strip():
                    cur.execute("INSERT INTO medications (user_id, name) VALUES (%s, %s)",
                                (user_id, medication.strip()))

            for allergy in allergies:
                if allergy.strip():
                    cur.execute("INSERT INTO allergies (user_id, name) VALUES (%s, %s)",
                                (user_id, allergy.strip()))

            for illness in illnesses:
                if illness.strip():
                    cur.execute("INSERT INTO illnesses (user_id, name) VALUES (%s, %s)",
                                (user_id, illness.strip()))

            mysql.connection.commit()
            flash('You have successfully signed up!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            mysql.connection.rollback()
            logging.error(f"Error in signup: {e}")
            flash(f'Signup failed. Please try again.', 'error')
            return redirect(url_for('signup'))
        finally:
            cur.close()

    # If GET request, just render the signup form
    return render_template('signup.html')

@app.route('/check_email', methods=['POST'])
def check_email():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", [email])
    user = cur.fetchone()
    cur.close()

    return jsonify({'exists': bool(user)})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    
    cur.execute("""
        SELECT *, 
        CASE 
            WHEN height > 0 AND weight > 0 
            THEN ROUND(weight / ((height/100) * (height/100)), 2)
            ELSE NULL
        END as bmi
        FROM users WHERE id = %s
    """, [user_id])
    user = cur.fetchone()
    
    cur.execute("SELECT name, description FROM medications WHERE user_id = %s", [user_id])
    medications = cur.fetchall()
    
    cur.execute("SELECT name, description FROM allergies WHERE user_id = %s", [user_id])
    allergies = cur.fetchall()
    
    cur.execute("SELECT name, description FROM illnesses WHERE user_id = %s", [user_id])
    illnesses = cur.fetchall()
    
    cur.execute("SELECT description FROM family_history WHERE user_id = %s", [user_id])
    family_history = cur.fetchone()
    
    cur.close()
    
    return render_template('dashboard.html', 
                           user=user,
                           medications=medications,
                           allergies=allergies,
                           illnesses=illnesses,
                           family_history=family_history['description'] if family_history else None)

@app.route('/symptom_checker')
@check_subscription_required
def symptom_checker():
    user_id = session['user_id']
    
    # Fetch user details from the users table
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT age, gender, height, weight, smoking_status, 
               exercise_habits, dietary_restrictions
        FROM users WHERE id = %s
    """, [user_id])
    user_details = cur.fetchone()
    cur.close()
    
    medical_history = get_user_medical_history(user_id)
    
    return render_template('symptom-checker.html', 
                           user_details=user_details, 
                           medical_history=medical_history)

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    symptoms = [
        "Abdominal pain", "Anxiety", "Back pain", "Bleeding", "Chest pain",
        "Cough", "Diarrhea", "Difficulty swallowing", "Dizziness", "Fatigue",
        "Fever", "Headache", "Heart palpitations", "Joint pain", "Nausea",
        "Neck pain", "Numbness", "Shortness of breath", "Skin problems", "Sore throat",
        "Urination problems", "Vision problems", "Vomiting", "Weakness"
    ]
    return jsonify(symptoms)

@app.route('/get_questions', methods=['POST'])
def get_questions():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        answered_questions = data.get('answered_questions', [])
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        # Fetch user details
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT age, gender, height, weight, smoking_status, 
                   exercise_habits, dietary_restrictions
            FROM users WHERE id = %s
        """, [user_id])
        user_details = cur.fetchone()
        cur.close()

        if not user_details:
            return jsonify({"error": "User details not found"}), 404

        medical_history = get_user_medical_history(user_id)

        # Generate follow-up questions using Groq AI
        questions = generate_followup_questions(symptoms, answered_questions, user_details, medical_history)

        if not questions:
            return jsonify({"error": "Unable to generate follow-up questions"}), 500

        # If we've asked enough questions, return an empty list to trigger diagnosis
        if len(answered_questions) >= 7:
            return jsonify([])

        # Return all generated questions
        return jsonify(questions)

    except Exception as e:
        app.logger.error(f"Error in get_questions: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

def generate_followup_questions(symptoms, answered_questions, user_details, medical_history):
    try:
        prompt = f"""
        Based on the following information, generate 1-3 relevant follow-up questions:

        User Details:
        - Age: {user_details['age']}
        - Gender: {user_details['gender']}
        - Height: {user_details['height']} cm
        - Weight: {user_details['weight']} kg
        - Smoking Status: {user_details['smoking_status']}
        - Exercise Habits: {user_details['exercise_habits']}
        - Dietary Restrictions: {user_details['dietary_restrictions']}

        Symptoms: {', '.join(symptoms)}
        Already asked questions: {', '.join(answered_questions)}
        Medical History:
        - Medications: {', '.join([med['name'] for med in medical_history['medications']])}
        - Allergies: {', '.join([allergy['name'] for allergy in medical_history['allergies']])}
        - Illnesses: {', '.join([illness['name'] for illness in medical_history['illnesses']])}

        Generate follow-up questions that help narrow down the possible conditions based on the symptoms and user information. 
        Do not repeat any previously asked questions.
        If this is the first symptom, ask about its duration, severity, or any accompanying symptoms.
        For subsequent questions, focus on related symptoms, risk factors, or lifestyle factors that could be relevant.

        Provide the questions in a JSON format like this:
        [
            {{"text": "Question 1 text here?", "options": ["Option 1", "Option 2", "Option 3"]}},
            {{"text": "Question 2 text here?", "options": ["Yes", "No"]}},
            {{"text": "Question 3 text here?", "options": ["Option A", "Option B", "Option C"]}}
        ]
        """

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant, expert in symptom analysis and diagnosis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        questions = json.loads(response.choices[0].message.content.strip())

        # Validate the structure of the questions
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Invalid question structure: empty list")
        
        for question in questions:
            if 'text' not in question or 'options' not in question:
                raise ValueError(f"Invalid question structure: {question}")

        return questions

    except json.JSONDecodeError as e:
        app.logger.error(f"Error decoding JSON in generate_followup_questions: {str(e)}")
        app.logger.error(f"Raw response: {response.choices[0].message.content}")
        return None
    except ValueError as e:
        app.logger.error(f"Error in generate_followup_questions: {str(e)}")
        return None
    except Exception as e:
        app.logger.error(f"Unexpected error in generate_followup_questions: {str(e)}")
        app.logger.error(traceback.format_exc())
        return None
            
@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        answers = data.get('answers', {})
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        # Fetch user details
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT age, gender, height, weight, smoking_status, 
                   exercise_habits, dietary_restrictions
            FROM users WHERE id = %s
        """, [user_id])
        user_details = cur.fetchone()
        cur.close()

        if not user_details:
            return jsonify({"error": "User details not found"}), 404

        medical_history = get_user_medical_history(user_id)

        # Generate diagnosis using Groq AI
        diagnosis = generate_diagnosis(symptoms, answers, user_details, medical_history)

        if diagnosis.get("error"):
            return jsonify({"error": diagnosis["error"]}), 500

        # Save the diagnosis to the database
        save_diagnosis(user_id, symptoms, answers, diagnosis)

        return jsonify(diagnosis)

    except Exception as e:
        app.logger.error(f"Error in diagnose route: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred while generating the diagnosis. Please try again later."}), 500

def generate_diagnosis(symptoms, answers, user_details, medical_history):
    try:
        prompt = f"""
        Based on the following information, provide a possible diagnosis:

        User Details:
        - Age: {user_details['age']}
        - Gender: {user_details['gender']}
        - Height: {user_details['height']} cm
        - Weight: {user_details['weight']} kg
        - Smoking Status: {user_details['smoking_status']}
        - Exercise Habits: {user_details['exercise_habits']}
        - Dietary Restrictions: {user_details['dietary_restrictions']}

        Symptoms: {', '.join(symptoms)}
        Answers to questions: {json.dumps(answers)}
        Medical History:
        - Medications: {', '.join([med['name'] for med in medical_history['medications']])}
        - Allergies: {', '.join([allergy['name'] for allergy in medical_history['allergies']])}
        - Illnesses: {', '.join([illness['name'] for illness in medical_history['illnesses']])}

        Provide the diagnosis in a JSON format like this:
        {{
            "diagnoses": [
                {{
                    "diagnosis": "Possible condition",
                    "probability": 0.8,
                    "description": "Brief description of the condition",
                    "common_medications": ["Medication 1", "Medication 2"],
                    "related_conditions": ["Related condition 1", "Related condition 2"]
                }},
                ...
            ],
            "symptom_relations": [
                "Symptom X may be related to your existing condition Y",
                ...
            ],
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2",
                ...
            ]
        }}
        """

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant, expert in symptom analysis and diagnosis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        diagnosis = json.loads(response.choices[0].message.content.strip())
        
        # Validate the structure of the diagnosis
        if not isinstance(diagnosis, dict) or "diagnoses" not in diagnosis:
            raise ValueError("Invalid diagnosis structure")

        return diagnosis

    except json.JSONDecodeError as e:
        app.logger.error(f"Error decoding JSON in generate_diagnosis: {str(e)}")
        app.logger.error(f"Raw response: {response.choices[0].message.content}")
        return {"error": "Error processing the diagnosis. Please try again."}
    except ValueError as e:
        app.logger.error(f"Error in generate_diagnosis: {str(e)}")
        return {"error": "Invalid diagnosis format. Please try again."}
    except Exception as e:
        app.logger.error(f"Unexpected error in generate_diagnosis: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {"error": "An unexpected error occurred while generating the diagnosis. Please try again later."}

def save_diagnosis(user_id, symptoms, answers, diagnosis):
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO user_diagnoses 
            (user_id, symptoms, answers, diagnosis_data, created_at) 
            VALUES (%s, %s, %s, %s, %s)
        """, (
            user_id,
            json.dumps(symptoms),
            json.dumps(answers),
            json.dumps(diagnosis),
            datetime.now()
        ))
        mysql.connection.commit()
        cur.close()
    except Exception as e:
        app.logger.error(f"Error saving diagnosis: {str(e)}")
        # Note: We're not raising an exception here to avoid interrupting the user experience
        # But you might want to handle this error differently based on your requirements


def extract_and_store_diagnoses():
    cur = mysql.connection.cursor()
    
    # Fetch diagnosis data from the user_diagnoses table
    cur.execute("SELECT id, diagnosis_data FROM user_diagnoses")
    rows = cur.fetchall()

    # List to store extracted diagnoses
    extracted_diagnoses = []

    for row in rows:
        user_id, diagnosis_data = row
        try:
            # Attempt to load the JSON data
            data = json.loads(diagnosis_data)

            # Extract the diagnoses
            for diagnosis in data['diagnoses']:
                extracted_diagnoses.append((user_id, diagnosis['diagnosis']))

        except json.JSONDecodeError:
            # Log or handle malformed JSON data if necessary
            app.logger.error(f"Invalid JSON format for user_id {user_id}")
            continue

   

    # Insert diagnoses into the extracted_diagnoses table
    for user_id, diagnosis in extracted_diagnoses:
        try:
            cur.execute("""
                INSERT IGNORE INTO extracted_diagnoses (user_id, diagnosis)
                VALUES (%s, %s)
            """, (user_id, diagnosis))  # Use IGNORE to skip duplicates
        except Exception as e:
            app.logger.error(f"Error inserting diagnosis: {e}")
            continue

    # Commit the changes
    mysql.connection.commit()
    
    # Close the cursor
    cur.close()

    app.logger.info("Diagnoses extraction and insertion complete.")




@app.route('/extract_diagnoses', methods=['GET'])
def extract_diagnoses_route():
    extract_and_store_diagnoses()
    return "Diagnoses extracted and stored successfully!"


@app.route('/get_past_diagnoses', methods=['GET'])
def get_past_diagnoses():
    if 'user_id' not in session:
        return jsonify({"error": "User not authenticated"}), 401

    user_id = session['user_id']

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, symptoms, diagnosis_data, created_at 
            FROM user_diagnoses 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, [user_id])
        diagnoses = cur.fetchall()
        cur.close()

        return jsonify([{
            'id': d['id'],
            'symptoms': json.loads(d['symptoms']),
            'diagnosis': json.loads(d['diagnosis_data']),
            'date': d['created_at'].isoformat()
        } for d in diagnoses])

    except Exception as e:
        app.logger.error(f"Error fetching past diagnoses: {str(e)}")
        return jsonify({"error": "An error occurred while fetching past diagnoses"}), 500

@app.route('/upload_report', methods=['POST'])
def upload_report():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    files = request.files.getlist('file')

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from PDF
                if filename.lower().endswith('.pdf'):
                    extracted_text = extract_text_from_pdf(file_path)
                else:
                    extracted_text = ""  # For non-PDF files, we're not extracting text
                
                # Save file information to database
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO medical_reports (user_id, filename, file_path, extracted_text) VALUES (%s, %s, %s, %s)",
                            (session['user_id'], filename, file_path, extracted_text))
                mysql.connection.commit()
                cur.close()

                uploaded_files.append(filename)
            except Exception as e:
                logging.error(f"Error processing file {file.filename}: {str(e)}")
                return jsonify({'success': False, 'message': f'Error processing file {file.filename}'}), 500

    if uploaded_files:
        return jsonify({'success': True, 'message': 'Files uploaded and text extracted successfully', 'files': uploaded_files})
    else:
        return jsonify({'success': False, 'message': 'No valid files were uploaded'}), 400

@app.route('/generate_summary', methods=['GET'])
def generate_summary():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        cur.execute("SELECT * FROM users WHERE id = %s", [user_id])
        user = cur.fetchone()
        
        cur.execute("SELECT name FROM medications WHERE user_id = %s LIMIT 5", [user_id])
        medications = [row['name'] for row in cur.fetchall()]
        
        cur.execute("SELECT name FROM allergies WHERE user_id = %s LIMIT 5", [user_id])
        allergies = [row['name'] for row in cur.fetchall()]
        
        cur.execute("SELECT name FROM illnesses WHERE user_id = %s LIMIT 5", [user_id])
        illnesses = [row['name'] for row in cur.fetchall()]
        
        cur.execute("SELECT id, filename, extracted_text, report_type FROM medical_reports WHERE user_id = %s ORDER BY id DESC LIMIT 5", [user_id])
        reports = cur.fetchall()
        
        cur.close()
        
        summary = {
            'personal_info': f"Name: {user['name']}<br>Age: {user['age']}<br>Gender: {user['gender']}",
            'medications': ", ".join(medications) if medications else "None",
            'allergies': ", ".join(allergies) if allergies else "None",
            'illnesses': ", ".join(illnesses) if illnesses else "None",
            'individual_reports': [],
            'consolidated_summary': ''
        }
        
        all_findings = []
        for report in reports:
            date, doctor, findings = extract_info(report['extracted_text'])
            report_summary = f"<strong>{report['report_type']}</strong> ({date})<br>"
            report_summary += f"Doctor: {doctor}<br>"
            report_summary += "Key Findings: " + "; ".join(findings) if findings else "No significant findings"
            
            summary['individual_reports'].append({
                'filename': report['filename'],
                'summary': report_summary
            })
            all_findings.extend(findings)
        
        # Generate consolidated summary
        consolidated_summary = "Key Observations Across Recent Reports:<br>"
        consolidated_summary += "<br>".join([f"- {finding}" for finding in set(all_findings[:5])])  # Limit to top 5 unique findings
        summary['consolidated_summary'] = consolidated_summary
        
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return jsonify({'success': False, 'message': 'Error generating summary'}), 500

@app.route('/add_item', methods=['POST'])
def add_item():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    user_id = session['user_id']
    data = request.json
    item_type = data.get('type')
    item_name = data.get('name')

    if not item_type or not item_name:
        return jsonify({'success': False, 'message': 'Invalid data'}), 400

    description = get_description(item_name, 'medications' if item_type in ['medication', 'allergy'] else 'diseases')

    cur = mysql.connection.cursor()
    try:
        if item_type == 'medication':
            cur.execute("INSERT INTO medications (user_id, name, description) VALUES (%s, %s, %s)", (user_id, item_name, description))
        elif item_type == 'allergy':
            cur.execute("INSERT INTO allergies (user_id, name, description) VALUES (%s, %s, %s)", (user_id, item_name, description))
        elif item_type == 'illness':
            cur.execute("INSERT INTO illnesses (user_id, name, description) VALUES (%s, %s, %s)", (user_id, item_name, description))
        else:
            return jsonify({'success': False, 'message': 'Invalid item type'}), 400

        mysql.connection.commit()
        return jsonify({'success': True, 'message': f'{item_type.capitalize()} added successfully', 'description': description})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()

@app.route('/get_descriptions', methods=['POST'])
def get_descriptions():
    data = request.json
    items = data.get('items', [])
    category = data.get('category', 'medications')
    
    descriptions = []
    for item in items:
        description = get_description(item, category)
        descriptions.append({'name': item, 'description': description})
    
    return jsonify({'descriptions': descriptions})

@app.route('/get_reports', methods=['GET'])
def get_reports():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, filename, report_type FROM medical_reports WHERE user_id = %s", [session['user_id']])
        reports = cur.fetchall()
        cur.close()
        return jsonify({'success': True, 'reports': reports})
    except Exception as e:
        logging.error(f"Error fetching reports: {str(e)}")
        return jsonify({'success': False, 'message': 'Error fetching reports'}), 500

@app.route('/get_report_content/<int:report_id>', methods=['GET'])
def get_report_content(report_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT extracted_text, report_type FROM medical_reports WHERE id = %s AND user_id = %s", [report_id, session['user_id']])
        report = cur.fetchone()
        cur.close()
        
        if report:
            return jsonify({'success': True, 'content': report['extracted_text'], 'report_type': report['report_type']})
        else:
            return jsonify({'success': False, 'message': 'Report not found'}), 404
    except Exception as e:
        logging.error(f"Error fetching report content: {str(e)}")
        return jsonify({'success': False, 'message': 'Error fetching report content'}), 500

@app.route('/delete_item', methods=['POST'])
def delete_item():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401

    user_id = session['user_id']
    data = request.json
    item_type = data.get('type')
    item_id = data.get('id')

    if not item_type or not item_id:
        return jsonify({'success': False, 'message': 'Invalid data'}), 400

    cur = mysql.connection.cursor()
    try:
        if item_type == 'medication':
            cur.execute("DELETE FROM medications WHERE user_id = %s AND name = %s", (user_id, item_id))
        elif item_type == 'allergy':
            cur.execute("DELETE FROM allergies WHERE user_id = %s AND name = %s", (user_id, item_id))
        elif item_type == 'illness':
            cur.execute("DELETE FROM illnesses WHERE user_id = %s AND name = %s", (user_id, item_id))
        elif item_type == 'report':
            cur.execute("DELETE FROM medical_reports WHERE user_id = %s AND id = %s", (user_id, item_id))
            # You might want to also delete the file from the server here
        else:
            return jsonify({'success': False, 'message': 'Invalid item type'}), 400

        mysql.connection.commit()
        return jsonify({'success': True, 'message': f'{item_type.capitalize()} deleted successfully'})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

# Diet Planner Routes

@app.route('/diet_planner')
@check_subscription_required
def diet_planner():
    return render_template('diet-planner.html')

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    data = request.json

    try:
        # Fetch user details
        cur = mysql.connection.cursor()
        cur.execute("SELECT age, gender, height, weight FROM users WHERE id = %s", [user_id])
        user_details = cur.fetchone()
        cur.close()

        if not user_details:
            return jsonify({'error': 'User details not found'}), 404

        # Prepare the prompt for Groq API
        prompt = f"""
        Generate a personalized diet plan based on the following user information:

        Age: {user_details['age']}
        Gender: {user_details['gender']}
        Height: {user_details['height']} cm
        Weight: {user_details['weight']} kg
        Goal: {data.get('goal', 'maintain weight')}
        Dietary Restrictions: {data.get('dietary_restrictions', 'None')}
        Preferred Cuisines: {', '.join(data.get('cuisines', ['Any']))}
        Meals per Day: {data.get('meals_per_day', '3 meals')}
        Activity Level: {data.get('physical_activity', 'moderately active')}
        Nutrient Requirements: {data.get('nutrient_requirements', 'balanced')}

        Please provide a detailed diet plan including:
        1. Daily calorie target
        2. Macronutrient breakdown (protein, carbs, fat percentages)
        3. A meal plan for one day with 3-5 meals (depending on meals per day preference)
        4. Total macronutrients and calories for the day

        Format the response as a JSON object with the following structure:
        {{
            "target_calories": 2000,
            "target_macros": {{
                "protein": 25,
                "carbs": 50,
                "fat": 25
            }},
            "diet_plan": [
                {{
                    "name": "Breakfast",
                    "meal": "Detailed meal description",
                    "calories": 500,
                    "protein": 20,
                    "carbs": 60,
                    "fat": 15
                }},
                ...
            ],
            "total_calories": 2000,
            "total_protein": 125,
            "total_carbs": 250,
            "total_fat": 55
        }}
        """

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a professional nutritionist, expert in creating personalized diet plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        diet_plan = response.choices[0].message.content.strip()

        # Check if the response starts with HTML
        if diet_plan.lower().startswith('<!doctype') or diet_plan.lower().startswith('<html'):
            app.logger.error(f"Received HTML instead of JSON: {diet_plan[:100]}...")
            return jsonify({"error": "Received an invalid response from the server. Please try again later."}), 500

        # Ensure the response is valid JSON
        try:
            diet_plan_json = json.loads(diet_plan)
        except json.JSONDecodeError as e:
            # If it's not valid JSON, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', diet_plan, re.DOTALL)
            if json_match:
                try:
                    diet_plan_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    app.logger.error(f"Failed to extract valid JSON: {diet_plan}")
                    return jsonify({"error": "Failed to parse diet plan. Please try again."}), 500
            else:
                app.logger.error(f"Invalid JSON response from AI model: {diet_plan}")
                return jsonify({"error": "Failed to generate a valid diet plan. Please try again."}), 500

        # Save the diet plan to the database
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO diet_plans (user_id, plan_data) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE plan_data = %s
        """, (user_id, json.dumps(diet_plan_json), json.dumps(diet_plan_json)))
        mysql.connection.commit()
        cur.close()

        return jsonify(diet_plan_json)

    except Exception as e:
        app.logger.error(f"Error generating diet plan: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred while generating the diet plan: {str(e)}. Please try again later."}), 500

def generate_meal_plan(target_calories, protein_percent, carbs_percent, fat_percent, dietary_restrictions, preferred_cuisines, meals_per_day):
    meal_plan = []
    remaining_calories = target_calories

    num_meals = 3 if '3 meals' in meals_per_day else 5

    for i in range(num_meals):
        meal_calories = remaining_calories / (num_meals - i)
        meal_protein = (meal_calories * protein_percent) / 4
        meal_carbs = (meal_calories * carbs_percent) / 4
        meal_fat = (meal_calories * fat_percent) / 9

        meal = get_meal(meal_calories, meal_protein, meal_carbs, meal_fat, dietary_restrictions, preferred_cuisines)
        meal_plan.append(meal)

        remaining_calories -= meal['calories']

    return meal_plan

def get_meal(target_calories, target_protein, target_carbs, target_fat, dietary_restrictions, preferred_cuisines):
    # This is a simplified version. In a real application, you'd have a more sophisticated meal selection algorithm.
    suitable_meals = [meal for meal in DIET_PLANS if
                      abs(meal['calories'] - target_calories) < 100 and
                      meal['protein'] >= target_protein * 0.8 and
                      meal['carbs'] >= target_carbs * 0.8 and
                      meal['fat'] >= target_fat * 0.8 and
                      all(restriction.lower() not in meal['meal'].lower() for restriction in dietary_restrictions.split(', ')) and
                      any(cuisine.lower() in meal['meal'].lower() for cuisine in preferred_cuisines)]

    if suitable_meals:
        return random.choice(suitable_meals)
    else:
        # If no suitable meal is found, return a generic meal
        return {
            'name': 'Custom Meal',
            'meal': 'Balanced meal based on your requirements',
            'calories': round(target_calories),
            'protein': round(target_protein),
            'carbs': round(target_carbs),
            'fat': round(target_fat)
        }

@app.route('/save_diet_plan', methods=['POST'])
def save_diet_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    plan_data = request.json

    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO diet_plans (user_id, plan_data) VALUES (%s, %s) ON DUPLICATE KEY UPDATE plan_data = %s",
                    (user_id, json.dumps(plan_data), json.dumps(plan_data)))
        mysql.connection.commit()
        cur.close()
        return jsonify({'success': True, 'message': 'Diet plan saved successfully'})
    except Exception as e:
        logging.error(f"Error saving diet plan: {str(e)}")
        return jsonify({'error': 'Failed to save diet plan'}), 500

@app.route('/get_saved_diet_plan', methods=['GET'])
def get_saved_diet_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT plan_data FROM diet_plans WHERE user_id = %s", [user_id])
        result = cur.fetchone()
        cur.close()

        if result:
            return jsonify(json.loads(result['plan_data']))
        else:
            return jsonify({'error': 'No saved diet plan found'}), 404
    except Exception as e:
        logging.error(f"Error retrieving diet plan: {str(e)}")
        return jsonify({'error': 'Failed to retrieve diet plan'}), 500

@app.route('/save_tracking', methods=['POST'])
def save_tracking():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    tracking_data = request.json

    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO diet_tracking (user_id, tracking_data) VALUES (%s, %s) ON DUPLICATE KEY UPDATE tracking_data = %s",
                    (user_id, json.dumps(tracking_data), json.dumps(tracking_data)))
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'Tracking data saved successfully'})
    except Exception as e:
        logging.error(f"Error saving tracking data: {str(e)}")
        return jsonify({'error': 'Failed to save tracking data'}), 500

@app.route('/get_tracking', methods=['GET'])
def get_tracking():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT tracking_data FROM diet_tracking WHERE user_id = %s", [user_id])
        result = cur.fetchone()
        cur.close()

        if result:
            return jsonify(json.loads(result['tracking_data']))
        else:
            return jsonify({})
    except Exception as e:
        logging.error(f"Error retrieving tracking data: {str(e)}")
        return jsonify({'error': 'Failed to retrieve tracking data'}), 500

# Exercise Planner Routes

@app.route('/exercise_planner')
@check_subscription_required
def exercise_planner():
    # Check if the user has an existing exercise plan
    cur = mysql.connection.cursor()
    cur.execute("SELECT plan_data FROM exercise_plans WHERE user_id = %s", [session['user_id']])
    existing_plan = cur.fetchone()
    cur.close()

    return render_template('exercise-planner.html', has_existing_plan=bool(existing_plan))

@app.route('/generate_exercise_plan', methods=['POST'])
def generate_exercise_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    user_responses = request.json

    # Fetch user details
    cur = mysql.connection.cursor()
    cur.execute("SELECT age, gender, height, weight FROM users WHERE id = %s", [user_id])
    user_details = cur.fetchone()

    if not user_details:
        cur.close()
        return jsonify({'error': 'User details not found'}), 404

    # Combine user details with responses
    prompt = f"""
    Generate a personalized exercise plan based on the following user information:

    Age: {user_details['age']}
    Gender: {user_details['gender']}
    Height: {user_details['height']} cm
    Weight: {user_details['weight']} kg
    Fitness Level: {user_responses["How would you describe your current fitness level?"]}
    Available Days: {user_responses["How many days per week can you commit to exercising?"]}
    Session Duration: {user_responses["How much time can you dedicate to each workout session?"]}
    Fitness Goals: {user_responses["What are your primary fitness goals? (Select up to 2)"]}
    Preferred Exercises: {user_responses["Which types of exercises do you enjoy or are interested in trying? (Select all that apply)"]}
    Exercise Location: {user_responses["Where do you prefer to exercise?"]}
    Available Equipment: {user_responses["What exercise equipment do you have access to? (Select all that apply)"]}
    Health Conditions: {user_responses["Do you have any health conditions or injuries that may affect your exercise routine?"]}
    Motivation Level: {user_responses["How would you rate your current motivation to start and stick to an exercise routine?"]}
    Routine Preference: {user_responses["Do you prefer structured workouts or more flexible routines?"]}

    Please provide a detailed exercise plan including:
    1. A brief introduction
    2. A weekly schedule
    3. Detailed exercise descriptions with sets, reps, and duration
    4. Additional recommendations

    Format the response as a JSON object with the following structure:
    {{
        "introduction": "Brief introduction text",
        "weeklySchedule": ["Day 1: ...", "Day 2: ...", ...],
        "exerciseDetails": [
            {{
                "name": "Exercise Name",
                "description": "Exercise description",
                "sets": "Number of sets",
                "reps": "Number of reps",
                "duration": "Duration if applicable"
            }},
            ...
        ],
        "additionalRecommendations": "Additional recommendations text"
    }}
    """

    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a professional fitness trainer, expert in creating personalized exercise plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        exercise_plan = response.choices[0].message.content.strip()

        # Check if the response starts with HTML
        if exercise_plan.lower().startswith('<!doctype') or exercise_plan.lower().startswith('<html'):
            app.logger.error(f"Received HTML instead of JSON: {exercise_plan[:100]}...")
            return jsonify({"error": "Received an invalid response from the server. Please try again later."}), 500

        # Ensure the response is valid JSON
        try:
            exercise_plan_json = json.loads(exercise_plan)
        except json.JSONDecodeError as e:
            # If it's not valid JSON, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', exercise_plan, re.DOTALL)
            if json_match:
                try:
                    exercise_plan_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    app.logger.error(f"Failed to extract valid JSON: {exercise_plan}")
                    return jsonify({"error": "Failed to parse exercise plan. Please try again."}), 500
            else:
                app.logger.error(f"Invalid JSON response from AI model: {exercise_plan}")
                return jsonify({"error": "Failed to generate a valid exercise plan. Please try again."}), 500

        exercise_plan_json['startDate'] = datetime.now().strftime('%Y-%m-%d')

        # Save the exercise plan to the database
        cur.execute("""
            INSERT INTO exercise_plans (user_id, plan_data) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE plan_data = %s
        """, (user_id, json.dumps(exercise_plan_json), json.dumps(exercise_plan_json)))
        mysql.connection.commit()

        cur.close()
        return jsonify(exercise_plan_json)

    except Exception as e:
        cur.close()
        app.logger.error(f"Error generating exercise plan: {str(e)}")
        return jsonify({"error": f"Failed to generate exercise plan: {str(e)}"}), 500

@app.route('/get_exercise_plan', methods=['GET'])
def get_exercise_plan():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']

    cur = mysql.connection.cursor()
    cur.execute("SELECT plan_data FROM exercise_plans WHERE user_id = %s", [user_id])
    result = cur.fetchone()
    cur.close()

    if result:
        try:
            return jsonify(json.loads(result['plan_data']))
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid exercise plan data'}), 500
    else:
        return jsonify({'error': 'No exercise plan found'}), 404
    
@app.route('/get_exercise_report', methods=['GET'])
def get_exercise_report():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    report_type = request.args.get('type', 'weeklyAdherence')

    try:
        cur = mysql.connection.cursor()

        if report_type == 'weeklyAdherence':
            # Fetch weekly adherence data for the last 12 weeks
            cur.execute("""
                SELECT 
                    YEARWEEK(date) as week,
                    AVG(CASE WHEN completed = 1 THEN 1 ELSE 0 END) * 100 as adherence
                FROM exercise_tracking
                WHERE user_id = %s AND date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
                GROUP BY YEARWEEK(date)
                ORDER BY week DESC
            """, (user_id,))
            adherence_data = cur.fetchall()
            
            report_data = {
                'weeklyAdherence': [
                    {
                        'week': row['week'],
                        'adherence': float(row['adherence']),
                        'label': f"Week {row['week'][-2:]}: {row['adherence']:.1f}%"
                    }
                    for row in adherence_data
                ]
            }

        elif report_type == 'exerciseTypeDistribution':
            # Fetch exercise type distribution
            cur.execute("""
                SELECT 
                    exercise_type,
                    COUNT(*) as count
                FROM exercise_tracking
                WHERE user_id = %s AND completed = 1
                GROUP BY exercise_type
                ORDER BY count DESC
                LIMIT 10
            """, (user_id,))
            distribution_data = cur.fetchall()
            
            report_data = {
                'exerciseTypeDistribution': [
                    {
                        'type': row['exercise_type'],
                        'count': row['count'],
                        'label': f"{row['exercise_type']}: {row['count']} times"
                    }
                    for row in distribution_data
                ]
            }

        elif report_type == 'progressOverTime':
            # Fetch progress over time (e.g., weight lifted for strength exercises)
            cur.execute("""
                SELECT 
                    date,
                    exercise_type,
                    MAX(weight) as max_weight
                FROM exercise_tracking
                WHERE user_id = %s AND exercise_type IN ('Bench Press', 'Squat', 'Deadlift')
                GROUP BY date, exercise_type
                ORDER BY date
                LIMIT 30
            """, (user_id,))
            progress_data = cur.fetchall()
            
            report_data = {
                'progressOverTime': [
                    {
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'exercise': row['exercise_type'],
                        'weight': float(row['max_weight']),
                        'label': f"{row['date'].strftime('%m/%d')}: {row['exercise_type']} - {row['max_weight']}kg"
                    }
                    for row in progress_data
                ]
            }

        elif report_type == 'caloriesBurned':
            # Fetch calories burned data for the last 30 days
            cur.execute("""
                SELECT 
                    date,
                    SUM(calories_burned) as total_calories
                FROM exercise_tracking
                WHERE user_id = %s AND date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY date
                ORDER BY date
            """, (user_id,))
            calories_data = cur.fetchall()
            
            report_data = {
                'caloriesBurned': [
                    {
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'calories': int(row['total_calories']),
                        'label': f"{row['date'].strftime('%m/%d')}: {int(row['total_calories'])} cal"
                    }
                    for row in calories_data
                ]
            }

        else:
            return jsonify({'error': 'Invalid report type'}), 400

        cur.close()

        return jsonify(report_data)

    except Exception as e:
        app.logger.error(f"Error generating exercise report: {str(e)}")
        return jsonify({'error': f'Failed to generate exercise report: {str(e)}'}), 500

@app.route('/check_subscription')
def check_subscription():
    if 'user_id' not in session:
        return jsonify({'subscribed': False})
    
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT subscription_status, subscription_expiry FROM users WHERE id = %s", [user_id])
    user = cur.fetchone()
    cur.close()
    
    if user and user['subscription_status'] == 'subscribed' and user['subscription_expiry'] > datetime.now().date():
        return jsonify({'subscribed': True})
    else:
        return jsonify({'subscribed': False})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'}), 401
    
    user_id = session['user_id']
    data = request.json
    plan = data.get('plan')
    
    if plan == '1-month':
        expiry_date = datetime.now() + timedelta(days=30)
    elif plan == '3-months':
        expiry_date = datetime.now() + timedelta(days=90)
    elif plan == '1-year':
        expiry_date = datetime.now() + timedelta(days=365)
    else:
        return jsonify({'success': False, 'message': 'Invalid plan'}), 400
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET subscription_status = 'subscribed', subscription_expiry = %s WHERE id = %s",
                    (expiry_date.date(), user_id))
        mysql.connection.commit()
        cur.close()
        return jsonify({'success': True, 'message': 'Subscription updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/payment')
def payment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    plan = request.args.get('plan')
    if plan not in ['1-month', '3-months', '1-year']:
        return redirect(url_for('home'))
    return render_template('payment.html', plan=plan)


@app.route('/save_exercise_tracking', methods=['POST'])
def save_exercise_tracking():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']
    data = request.json

    if not data:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        cur = mysql.connection.cursor()
        
        # Check if a record already exists for this user
        cur.execute("SELECT tracking_data FROM exercise_tracking WHERE user_id = %s", [user_id])
        existing_data = cur.fetchone()

        if existing_data:
            # If data exists, update it
            tracking_data = json.loads(existing_data[0])
            tracking_data.update(data)
            cur.execute("""
                UPDATE exercise_tracking 
                SET tracking_data = %s, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
            """, (json.dumps(tracking_data), user_id))
        else:
            # If no data exists, insert new record
            cur.execute("""
                INSERT INTO exercise_tracking (user_id, tracking_data)
                VALUES (%s, %s)
            """, (user_id, json.dumps(data)))

        mysql.connection.commit()
        cur.close()
        return jsonify({'success': True, 'message': 'Exercise tracking data saved successfully'})
    except Exception as e:
        logging.error(f"Error saving exercise tracking data: {str(e)}")
        return jsonify({'error': 'Failed to save exercise tracking data'}), 500

@app.route('/get_exercise_tracking', methods=['GET'])
def get_exercise_tracking():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_id = session['user_id']

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT tracking_data FROM exercise_tracking WHERE user_id = %s", [user_id])
        result = cur.fetchone()
        cur.close()

        if result:
            tracking_data = json.loads(result[0])
            return jsonify(tracking_data)
        else:
            return jsonify({})
    except Exception as e:
        logging.error(f"Error retrieving exercise tracking data: {str(e)}")
        return jsonify({'error': 'Failed to retrieve exercise tracking data'}), 500

# Admin credentials (in a real-world scenario, these should be stored securely, e.g., hashed in a database)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('admin_login.html')
    elif request.method == 'POST':
        username = request.json.get('username')
        password = request.json.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'})

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    """Render the admin dashboard."""
    return render_template('admin.html')

@app.route('/api/admin/report/userActivity', methods=['GET'])
@login_required
def get_user_activity():
    try:
        cur = mysql.connection.cursor()

        # Get total users
        cur.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cur.fetchone()['total_users']

        # Get active users (users who logged in within the last 7 days)
        cur.execute("SELECT COUNT(*) as active_users FROM users WHERE last_login >= NOW() - INTERVAL 7 DAY")
        active_users = cur.fetchone()['active_users']

        cur.close()

        return jsonify({
            'totalUsers': total_users,
            'activeUsers': active_users
        })
    except Exception as e:
        app.logger.error(f"Error in User Activity Report: {e}")
        return jsonify({'error': 'Failed to fetch user activity'}), 500

@app.route('/api/admin/report/subscriptionStatus', methods=['GET'])
@login_required
def get_subscription_status():
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT COUNT(*) as count FROM users WHERE subscription_status = 'subscribed'")
        total_subscribed = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM users WHERE subscription_status = 'free'")
        total_free = cur.fetchone()['count']

        cur.close()

        return jsonify({
            'totalSubscribed': total_subscribed,
            'totalFree': total_free
        })
    except Exception as e:
        app.logger.error(f"Error in Subscription Status Report: {e}")
        return jsonify({'error': 'Failed to fetch subscription status'}), 500

@app.route('/api/admin/report/diagnoses', methods=['GET'])
@login_required
def get_diagnosis_report():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT diagnosis_data, COUNT(*) as count
            FROM user_diagnoses
            GROUP BY diagnosis_data
            ORDER BY count DESC
            LIMIT 5
        """)
        diagnoses = cur.fetchall()
        cur.close()

        report_data = []
        for diagnosis in diagnoses:
            diagnosis_json = json.loads(diagnosis['diagnosis_data'])
            if 'diagnoses' in diagnosis_json and len(diagnosis_json['diagnoses']) > 0:
                report_data.append({
                    'diagnosis': diagnosis_json['diagnoses'][0]['diagnosis'],
                    'count': diagnosis['count'],
                    'probability': diagnosis_json['diagnoses'][0]['probability']
                })

        return jsonify(report_data)
    except Exception as e:
        app.logger.error(f"Error in Diagnosis Report: {e}")
        return jsonify({'error': 'Failed to fetch diagnosis report'}), 500

@app.route('/api/admin/report/common-symptoms', methods=['GET'])
@login_required
def get_common_symptoms():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT symptoms, COUNT(*) as count
            FROM user_diagnoses
            GROUP BY symptoms
            ORDER BY count DESC
            LIMIT 10
        """)
        symptoms = cur.fetchall()
        cur.close()

        symptom_data = []
        for symptom in symptoms:
            symptom_list = json.loads(symptom['symptoms'])
            for s in symptom_list:
                symptom_data.append({
                    'symptom': s,
                    'count': symptom['count']
                })

        # Aggregate symptom counts
        symptom_counts = {}
        for item in symptom_data:
            if item['symptom'] in symptom_counts:
                symptom_counts[item['symptom']] += item['count']
            else:
                symptom_counts[item['symptom']] = item['count']

        # Sort and limit to top 10
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify([{'symptom': s[0], 'count': s[1]} for s in top_symptoms])
    except Exception as e:
        app.logger.error(f"Error in Common Symptoms Report: {e}")
        return jsonify({'error': 'Failed to fetch common symptoms'}), 500

@app.route('/api/admin/report/reportTypes', methods=['GET'])
@login_required
def get_report_types():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT report_type, COUNT(*) as count
            FROM medical_reports
            GROUP BY report_type
            ORDER BY count DESC
        """)
        report_types = cur.fetchall()
        cur.close()

        labels = [r['report_type'] for r in report_types]
        counts = [r['count'] for r in report_types]

        return jsonify({
            'labels': labels,
            'counts': counts
        })
    except Exception as e:
        app.logger.error(f"Error in Report Types Report: {e}")
        return jsonify({'error': 'Failed to fetch report types'}), 500

@app.route('/api/admin/users', methods=['GET'])
@login_required
def get_users():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, name, email, subscription_status,
                   CASE WHEN subscription_status = 'subscribed' THEN 'Active' ELSE 'Inactive' END as status,
                   CASE WHEN is_blocked = 0 THEN 'Unblocked' ELSE 'Blocked' END as block_status
            FROM users
            ORDER BY id DESC
            LIMIT 10
        """)
        users = cur.fetchall()
        cur.close()
        
        return jsonify(users)
    except Exception as e:
        app.logger.error(f"Error fetching users: {e}")
        return jsonify({'error': 'Failed to fetch users'}), 500

@app.route('/api/admin/users/toggle-block', methods=['POST'])
@login_required
def toggle_user_block():
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        cur = mysql.connection.cursor()
        
        # Check current block status
        cur.execute("SELECT is_blocked FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        new_status = 1 if user['is_blocked'] == 0 else 0
        
        # Update block status
        cur.execute("UPDATE users SET is_blocked = %s WHERE id = %s", (new_status, user_id))
        mysql.connection.commit()
        
        cur.close()
        
        return jsonify({'success': True, 'new_status': 'Blocked' if new_status == 1 else 'Unblocked'})
    except Exception as e:
        app.logger.error(f"Error toggling user block status: {e}")
        return jsonify({'error': 'Failed to update user status'}), 500


@app.route('/api/admin/test_db_connection')
@login_required
def test_db_connection():
    """API endpoint to test the database connection."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        
        if result:
            return jsonify({'status': 'success', 'message': 'Database connection successful'})
        else:
            return jsonify({'status': 'error', 'message': 'Unexpected database response'})
    except Exception as e:
        app.logger.error(f"MySQL Error in test_db_connection: {e}")
        return jsonify({'status': 'error', 'message': f'Database connection failed: {e}'})


@app.route('/save_weight_entry', methods=['POST'])
def save_weight_entry():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    user_id = session['user_id']
    date = request.json.get('date')
    weight = request.json.get('weight')
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO weight_entries (user_id, date, weight) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE weight = %s", 
                    (user_id, date, weight, weight))
        mysql.connection.commit()
        cur.close()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error saving weight entry: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_weight_data', methods=['GET'])
def get_weight_data():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    user_id = session['user_id']
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT target_weight FROM users WHERE id = %s", (user_id,))
        target_weight = cur.fetchone()['target_weight']
        
        cur.execute("SELECT date, weight FROM weight_entries WHERE user_id = %s ORDER BY date", (user_id,))
        weight_entries = cur.fetchall()
        
        weight_data = {entry['date'].isoformat(): entry['weight'] for entry in weight_entries}
        last_weight_update_date = max(weight_data.keys()) if weight_data else None
        
        cur.close()
        
        return jsonify({
            'success': True,
            'targetWeight': target_weight,
            'weightData': weight_data,
            'lastWeightUpdateDate': last_weight_update_date
        })
    except Exception as e:
        app.logger.error(f"Error getting weight data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)