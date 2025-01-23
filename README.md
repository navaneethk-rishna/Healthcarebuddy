
---

# Healthcare Buddy

**Healthcare Buddy** is an advanced healthcare management system designed to assist users with health monitoring, symptom checking, personalized diet and exercise plans, medication management, and more. This project leverages cutting-edge AI and machine learning technologies to provide an intuitive and interactive healthcare experience.

---

## Features

- **Symptom Checker**  
  Type in symptoms to receive suggestions and answer guided yes/no questions for potential diagnosis.  


 

- **Exercise Planner**  
  Personalized exercise plans tailored to user goals (e.g., weight gain, weight loss).  

- **Yoga Instruction Page**  
  Learn yoga techniques and routines for holistic well-being.  

- **Diet Plans**  
  Customized diet recommendations for weight management or specific health conditions.  

- **Scanning Reports**  
  Upload images of medical reports and extract text using OCR for better health tracking.  

- **User-Friendly Registration System**  
  Step-by-step signup process with fields for personal and medical information.

---

## Technologies Used

### Frontend
- **HTML**, **CSS**, **JavaScript**  
  - Interactive, responsive design for an engaging user experience.  

### Backend
- **Python Flask**  
  - Lightweight and efficient framework for managing backend logic.  

### Database
- **MySQL**  
  - Stores user information, health data, and medical records.  


### Hosting
- **WAMP Server**  
  - Hosts the MySQL database for backend operations.  

---

## Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/healthcare-buddy.git
   cd healthcare-buddy
   ```

  

2. **Database Configuration**  
   - Set up the database in PHPMyAdmin with the following configuration:  
     ```
     new mysqli('localhost', 'root', '', 'healthcare_buddy');
     ```

   - Import the provided SQL file to set up the necessary tables and sample data.

3. **Run the Application**  
   Start the Flask development server:  
   ```bash
   flask run
   ```

4. **Access the Application**  
   Open your browser and navigate to:  
   ```
   http://localhost:5000
   ```

---

## Project Structure

```
healthcare-buddy/
├── app.py                # Main Flask application
├── templates/            # HTML templates
│   ├── index.html
│   ├── signup.html
│   ├── ...
├── static/               # Static files (CSS, JS, images)
│   ├── style.css
│   ├── script.js
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── wsgi.py               # Application entry point for deployment
└── database.sql          # SQL file for database setup
```

---

## Contributions

We welcome contributions from the community! Please follow these steps:

1. Fork the repository.  
2. Create a new branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m "Add feature"`.  
4. Push to the branch: `git push origin feature-name`.  
5. Open a pull request.  

---



---


--- 

Let me know your GitHub username or any other details to include!
