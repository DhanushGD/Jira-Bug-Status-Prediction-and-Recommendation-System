import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

proxy = http_proxy=""  #your proxy (if needed)
nltk.set_proxy(proxy)
# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Fix for the LookupError

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the dataset
bug_data = pd.read_csv('jira_issues.csv', delimiter=";", on_bad_lines='skip')

# Handle missing values and map resolutions to fixed and invalid values
bug_data['resolution'].fillna("In_progress", inplace=True)
invalid_values = ["Won't Do", "Won't Fix", "Without Feedback", "Rejected", "Cancelled", "Cannot Reproduce", "Out of Scope"]
fixed_value = ["Done", "Delivered"]
bug_data.loc[bug_data['resolution'].isin(invalid_values), 'resolution'] = "Invalid"
bug_data.loc[bug_data['resolution'].isin(fixed_value), 'resolution'] = "Fixed"

# Extract bug descriptions (summaries) and resolutions (statuses)
bug_descriptions = bug_data['summary'].values.tolist()
statuses = bug_data['resolution'].values.tolist()

# Prepare the vocabulary, documents, and classes
words = []
classes_status = []
documents = []
ignore_words = ['?', '!', '[', ']']

for idx, (bug_description, status) in enumerate(zip(bug_descriptions, statuses)):
    w = nltk.word_tokenize(bug_description)  # Tokenize each description
    words.extend(w)  # Add to the words list
    documents.append((w, str(status)))  # Associate words with the status
    if str(status) not in classes_status:
        classes_status.append(str(status))  # Add the status to the list of classes

# Lemmatize words and remove unwanted characters
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # Sort and remove duplicates
classes_status = sorted(list(set(classes_status)))  # Sort and remove duplicates

# Save the processed words and classes to pickle files
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes_status, open('labels_status.pkl', 'wb'))

# Prepare training data
training = []
output_empty_status = [0] * len(classes_status)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]  # Lemmatize the words

    # Create the bag of words for each description
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output row for the status (one-hot encoding)
    output_row_status = list(output_empty_status)
    output_row_status[classes_status.index(doc[1])] = 1

    training.append([bag, output_row_status])

# Shuffle the training data
random.shuffle(training)

# Prepare training data for input and output
train_x = np.array([item[0] for item in training])
train_y_status = np.array([item[1] for item in training])

# Pad the sequences to ensure uniform input length
train_x = pad_sequences(train_x, padding='post')

# Build the model
model_status = Sequential()
model_status.add(Dense(5, input_shape=(train_x.shape[1],), activation='relu'))  # Input layer
model_status.add(Dropout(0.5))  # Dropout for regularization
model_status.add(Dense(5, activation='relu'))  # Hidden layer
model_status.add(Dropout(0.5))  # Dropout for regularization
model_status.add(Dense(len(classes_status), activation='softmax'))  # Output layer (one-hot encoding)

# Compile the model
sgd_status = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_status.compile(loss='categorical_crossentropy', optimizer=sgd_status, metrics=['accuracy'])

# Train the model with full epochs (no early stopping)
model_status.fit(train_x, train_y_status, epochs=200, batch_size=5, verbose=1, validation_split=0.2)

# Evaluate the model on training data
train_loss, train_accuracy = model_status.evaluate(train_x, train_y_status)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Split the data into training and test sets (80% training, 20% testing)
test_split_idx = int(0.8 * len(train_x))
X_test_full = train_x[test_split_idx:]
y_test_full = train_y_status[test_split_idx:]

# Evaluate the model on the test data
test_loss, test_accuracy = model_status.evaluate(X_test_full, y_test_full)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model_status.save('model_status.h5')

# Save the maximum sequence length (for future use)
max_length = train_x.shape[1]
with open('max_length.pkl', 'wb') as f:
    pickle.dump(max_length, f)

print("Model training completed and saved.")
