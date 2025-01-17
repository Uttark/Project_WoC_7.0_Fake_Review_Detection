import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Download NLTK data
nltk.download('stopwords')

# Load the dataset
file_path = 'C:/Users/Asus/Desktop/cleaned_amazon_reviews.csv'    # Specify location of file (use forward slash while giving path
data = pd.read_csv(file_path)

# Check if required columns exist
if 'text_' not in data.columns or 'label' not in data.columns:
    raise KeyError("Dataset must contain 'text_' and 'label' columns.")

# Preprocessing
data = data.dropna(subset=['text_', 'label'])  # Remove rows with missing data
data = data.drop_duplicates(subset=['text_'])  # Remove duplicate reviews

# Normalize text to lowercase
data['text_'] = data['text_'].str.lower()

# Remove punctuation and special characters
data['text_'] = data['text_'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['processed_text'] = data['text_'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Convert labels (CG: 1, OR: 0)
label_mapping = {'CG': 1, 'OR': 0}
data['label'] = data['label'].map(label_mapping)

# Ensure no invalid labels remain
if data['label'].isnull().any():
    raise ValueError("Labels contain invalid values. Only 'CG' and 'OR' are allowed.")

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['processed_text'])  # Features
y = data['label']                                    # Target labels

# Dataset Preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Train and Evaluate Models
results = {}
for name, model in models.items():
    pipeline = Pipeline([('model', model)])
    pipeline.fit(X_train, y_train)  # Train the model
    y_pred = pipeline.predict(X_test)  # Predictions
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)

# Select the best-performing model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = models[best_model_name]

# Serialize the best model
model_path = f'best_model_{best_model_name.replace(" ", "_").lower()}.joblib'
joblib.dump(best_model, model_path)

print(f"Best model ({best_model_name}) saved to {model_path}")
