from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import nltk
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import csv
import re
import os

# Flask app setup
app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set up the path to your downloads directory
DOWNLOAD_FOLDER = 'code_changes.dif'  # Path where your downloadable files are stored
app.config['UPLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Load models and data
model_status = load_model('model_status.h5',compile=False)

# Load the vocabulary (words) and class labels (status labels)
with open('texts.pkl', 'rb') as f:
    words = pickle.load(f)

with open('labels_status.pkl', 'rb') as f:
    classes_status = pickle.load(f)

# Load the maximum sequence length
with open('max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

# Load bug data
df1 = pd.read_csv('jira_issues.csv', delimiter=";", on_bad_lines='skip')
df2 = pd.read_csv("summarized_comments.csv")
df1['resolution'].fillna("In_progress", inplace=True)
invalid_values = ["Won't Do", "Won't Fix", "Without Feedback", "Rejected", "Cancelled", "Cannot Reproduce", "Out of Scope"]
fixed_value = ["Done", "Delivered"]
df1.loc[df1['resolution'].isin(invalid_values), 'resolution'] = "Invalid"
df1.loc[df1['resolution'].isin(fixed_value), 'resolution'] = "Fixed"
df1.fillna("Not Available", inplace=True)
bug_data = pd.merge(df1, df2, on='issue_id')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the bug description (tokenize and lemmatize)
def preprocess_description(description):
    tokens = nltk.word_tokenize(description)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

def is_repetitive_input(description):
    """
    Check if the description contains repetitive patterns like "23 23 23" or other simplistic patterns.
    """
    repeated_pattern = r"(\b\w+\b)(?:\s+\1)+"

    if re.match(repeated_pattern, description.strip()):
        return True
    return False


def predict_status(description):
    print("entered prediction")
    try:
        if not description:
            raise ValueError("Description cannot be empty")

        # Check for repetitive input patterns (e.g., "23 23 23 23")
        if is_repetitive_input(description):
            return "Invalid Input", [], [], [], "", "Please enter a more meaningful description."  # Return warning message

        # Preprocess the description (tokenize and lemmatize)
        tokens = preprocess_description(description)

        # Create a bag of words based on the description
        bag = [1 if w in tokens else 0 for w in words]
        padded_bag = pad_sequences([bag], maxlen=max_length, padding='post')

        # Get the model's predictions for bug status (resolution)
        predictions = model_status.predict(padded_bag)[0]
        status_index = np.argmax(predictions)
        status = classes_status[status_index]

        # Find most similar bugs using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer().fit(bug_data['summary'])
        input_vec = vectorizer.transform([description])
        description_vectors = vectorizer.transform(bug_data['summary'])
        similarities = cosine_similarity(input_vec, description_vectors).flatten()

        threshold = 0.5
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]

        # If similarity is greater than 0.9, mark the status as "Duplicate"
        if any(sim >= 0.9 for sim in similarities):
            status = "Duplicate"

        if len(filtered_indices) == 0:
            return "Invalid Input", [], [], [], "", "No similar bugs found."

        # Get the filtered bugs based on the cosine similarity threshold
        filtered_bugs = bug_data[['issue_id', 'summary', 'summarize_comment', 'fix_versions', 'resolution']].values[filtered_indices]

        # Jira root URL for constructing bug links
        root_link = ""  #your root jira link

        # Prepare the output data for the top 3 most similar bugs
        top_bugs = []
        top_jira_links = []
        top_summaries = []
        fix_versions = []
        comments = []
        for i in range(min(3, len(filtered_bugs))):  # Get top 3 bugs
            fix_versions.append(filtered_bugs[i][3])
            most_similar_bug = filtered_bugs[i].tolist()
            top_bugs.append(most_similar_bug[1])  # Bug description
            top_jira_links.append(root_link + str(most_similar_bug[0]))  # Generate Jira link using issue_id
            from Jira import fetch_data
            fetch_data(most_similar_bug[0])


            if 'Not Available' not in most_similar_bug[2]:
                top_summaries.append(most_similar_bug[2])  # Comments

            comments.append(f"Jira ID {most_similar_bug[0]}: {most_similar_bug[2]}")  # Including Jira ID in comments

        # Remove duplicate "Not Available" from Fix Version
        fix_versions = list(dict.fromkeys(fix_versions))

        if len(top_summaries) == 0:
            top_summaries = ["Not Available"]
        print(top_jira_links)
        return status, top_bugs, top_jira_links, top_summaries, fix_versions, ""  # Return empty string for no warning

    except Exception as e:
        print(f"Error in predict_status: {str(e)}")
        return "Error", [], [], [], "", ""  # Return an error message and empty lists for bugs, jira links, and comments


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        description = data.get('description', '')  # Retrieve the description

        if not description:
            return jsonify({
                'status': 'Invalid Input',
                'most_similar_bugs': [],
                'jira_links': [],
                'summarized_comments': [],
                'fix_version': '',
                'warning_message': 'Please enter a bug description before predicting.'
            })

        # Call predict_status to get the resolution, fix version, and similar bugs
        status, most_similar_bugs, jira_links, summarized_comments, fix_version, warning_message = predict_status(description)

        return jsonify({
            'status': status,  # Predicted resolution
            'most_similar_bugs': most_similar_bugs,
            'jira_links': jira_links,
            'summarized_comments': summarized_comments,
            'fix_version': fix_version,  # Return the fix version
            'warning_message': warning_message  # Return warning message if present
        })

    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({
            'status': 'invalid input.',
            'most_similar_bugs': [],
            'jira_links': [],
            'summarized_comments': [],
            'fix_version': '',
            'warning_message': 'An error occurred. Please try again later.'  
        })

# Add the following route to serve the file from the templates directory
@app.route('/download_report')
def download_report():
    return send_from_directory(
        os.path.join(app.root_path, 'templates'),  # Path to the 'templates' folder
        'code_changes.html',  # Name of the file you want to serve
        as_attachment=True  # This will trigger a file download in the browser
    )

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
