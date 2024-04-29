from flask import Flask, request, Response
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__,template_folder='.')

# Load the machine learning model
model_path = "salary_predictor_model.pkl"
model = joblib.load(model_path)

# Define function to make predictions
def predict_salary(input_data):
    # Make predictions using the loaded model
    prediction = model.predict(input_data)  # Ensure input_data is a list
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        job_title = request.form['jobTitle']
        age = int(request.form['age'])
        education_level = request.form['educationLevel']
        years_of_experience = int(request.form['experience'])

        # Preprocess the input data
        education_level_encoded = encode_education_level(education_level)
        job_title_encoded = encode_job_title(job_title)

        # Create input data for prediction
        input_data = [[age, education_level_encoded, job_title_encoded, years_of_experience]]

        # Make prediction using the model
        prediction = predict_salary(input_data)

        # Render the HTML file with prediction result
        with open(os.path.join(os.path.dirname(__file__), 'salary.html'), 'r') as file:
            html_content = file.read()
            html_content = html_content.replace('<!-- prediction_placeholder -->', f'<p>Prediction: {prediction[0]}</p>')
        return html_content
    
    # Render the HTML file without prediction result
    with open(os.path.join(os.path.dirname(__file__), 'salary.html'), 'r') as file:
        html_content = file.read()
    return html_content

def encode_education_level(education_level):
    # Write code here to encode education level
    # Example:
    if education_level == "Bachelor's":
        return 0
    elif education_level == "Master's":
        return 1
    elif education_level == "PhD":
        return 2
    else:
        return -1

def encode_job_title(job_title):
    # Write code here to encode job title
    # Example:
    if job_title == 'data analyst':
        return 0
    elif job_title == 'data scientist':
        return 1
    elif job_title == 'software engineer':
        return 2
    else:
        return -1


course_data = pd.read_csv('course 2.csv')

# Define TF-IDF vectorizer for course prediction
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))

# Fit TF-IDF vectorizer on course skills
X_tfidf = tfidf_vectorizer.fit_transform(course_data['skills'])

# Define function to recommend courses based on input skills
def recommend_courses_based_on_skills(input_skills, top_n=5):
    input_skills_tfidf = tfidf_vectorizer.transform([input_skills])
    input_cosine_sim = cosine_similarity(input_skills_tfidf, X_tfidf)
    avg_sim_scores = np.mean(input_cosine_sim, axis=0)
    top_courses_indices = np.argsort(avg_sim_scores)[::-1][:top_n]
    recommended_courses_info = course_data.iloc[top_courses_indices][['course', 'duration', 'level']]
    return recommended_courses_info

@app.route('/coursepred', methods=['POST'])
def course_recommendation():
    if 'skills' in request.form:
        input_skills = request.form['skills']
        recommended_courses_info = recommend_courses_based_on_skills(input_skills)
        courses_html = ""
        for index, row in recommended_courses_info.iterrows():
            courses_html += f"<p>{row['course']} - Duration: {row['duration']} - Level: {row['level']}</p>"
        return courses_html
    else:
        return "No input skills provided."

if __name__ == '__main__':
    app.run(debug=True)

