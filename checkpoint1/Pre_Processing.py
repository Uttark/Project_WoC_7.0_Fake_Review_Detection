import pandas as pd  
import re
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

# Load the dataset
file_path = 'C:/Users/Asus/Desktop/AMAZON REVIEW/scrapping/cleaned_amazon_reviews.csv'
data = pd.read_csv(file_path)

# Ensure 'Reviews' column exists
if 'Reviews' not in data.columns:
    raise KeyError("'Reviews' column not found in the dataset.")

# Drop rows with missing reviews and remove duplicates
data = data.dropna(subset=['Reviews'])
data = data.drop_duplicates(subset=['Reviews'])

# Normalize text to lowercase
data['Reviews'] = data['Reviews'].str.lower()

# Remove punctuation and special characters
data['Reviews'] = data['Reviews'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Ensure all reviews are strings
data['Reviews'] = data['Reviews'].astype(str)



stop_words = set(stopwords.words('english'))# Load stopwords
# Tokenize and remove stopwords
data['tokens'] = data['Reviews'].apply(lambda x: [word for word in x.split() if word not in stop_words])
print(data)


# Combine tokens back into text
data['processed_text'] = data['tokens'].apply(lambda x: ' '.join(x))

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the text data
X = vectorizer.fit_transform(data['processed_text'])

# Print the shape of the matrix
print(f"Vectorized matrix shape: {X.shape}")

# Get the feature names (vocabulary)
print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")