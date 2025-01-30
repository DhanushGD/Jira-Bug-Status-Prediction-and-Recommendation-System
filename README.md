# Jira-Bug-Status-Prediction-and-Recommendation-System

## Overview
The **Bug Status Prediction and Recommendation System** is an AI-powered tool designed to streamline bug tracking and debugging processes. It leverages deep learning models to predict bug statuses and provides smart recommendations for similar bugs and possible fixes. Integrated with Jira and GitLab, this tool helps developers and QA teams improve collaboration and speed up bug triaging.

## Key Features
- 🧠 **AI Predictions**: A deep learning model built with TensorFlow predicts the status of bugs based on their descriptions.
- 💡 **Smart Recommendations**: Utilizes TF-IDF and cosine similarity to suggest similar bugs and recommend fixes efficiently.
- 🔍 **Jira & GitLab Integration**:  
  - Scrapes GitLab links from Jira tickets using Beautiful Soup.  
  - Integrates with GitLab API to fetch commit details, code diffs, and file changes related to bugs.  
  - Allows developers to download changes in HTML format for offline reference.
- 🖥️ **Web App Interface**: A user-friendly web interface designed using Flask, HTML, CSS, and Bootstrap to ensure a smooth user experience.
- 🐳 **Dockerized Deployment**: The tool is containerized using Docker for easy and consistent deployment.

## Tech Stack
- **Python** 🐍
- **Flask** 🌐
- **TensorFlow** 🤖
- **Beautiful Soup** 🥣
- **GitLab API** 
- **HTML, CSS, JavaScript**
- **Docker**

## Impact
- Simplifies and speeds up the bug triaging process 🔥.
- Enhances debugging productivity by providing historical bug insights and direct access to code changes.
- Streamlines collaboration between developers 👨‍💻 and QA teams 👩‍💻.

## Installation

### Prerequisites
- Python 3.9 or above
- Docker (for containerized deployment)

### Steps to Run Locally
1. **Clone the repository**:
   ```bash
   https://github.com/DhanushGD/Jira-Bug-Status-Prediction-and-Recommendation-System.git
   cd To the file
2. **Install dependencies**: If you're running locally, install dependencies via pip:
   ```bash
   pip install -r requirements.txt
3. **Run the application**: If you are not using Docker, you can run the app using:
    ```bash
    python app.py
4. **Docker Deployment**:
   - Build the Docker image:
       ```bash
       docker build -t bug-prediction-tool .
   - Run the Docker container:
       ```bash
     docker run -p 5000:5000 bug-prediction-tool
     
The application will be accessible on http://localhost:5000.

### Usage

### 1. **Prepare Your Own Data (data.csv)**

Create or download a `data.csv` file with the following structure:

| bug_id | bug_description | status      | commit_id | file_changes | fix_suggestion |
|--------|-----------------|-------------|-----------|--------------|----------------|
| 1      | Bug desc 1       | In Progress | abc123    | file1.py     | Fix 1          |
| 2      | Bug desc 2       | Closed      | def456    | file2.py     | Fix 2          |

Each row should contain:
- **bug_id**: Unique ID
- **bug_description**: Text description of the bug
- **status**: Bug status (e.g., Open, In Progress, Closed)
- **commit_id**: GitLab commit ID
- **file_changes**: Modified files
- **fix_suggestion**: Suggested fix

### 2. **Summarize the Contents of Your `data.csv`**
Ensure that your `data.csv` file contains:
- Descriptive bug information for accurate predictions
- Consistent status labels (e.g., Open, In Progress, Closed)
- Correct commit IDs that correspond to relevant GitLab data
- Accurate file changes and potential fixes

### 3. **Train the Model: Run the training script to process your data.csv and train the AI model:**
```bash
python training.py
```

After running the app, navigate to http://localhost:5000 in your browser.
The interface allows you to input bug descriptions, view the predicted status, and get recommended fixes based on AI predictions.
You can also search for similar bugs and view commit details fetched from GitLab related to the selected bug.

