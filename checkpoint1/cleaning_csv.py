import pandas as pd

# Load the CSV file into a DataFrame
input_file = "amazon_reviews.csv"  # Replace with the path to your CSV file
output_file = "cleaned_amazon_reviews.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Drop rows where any of the required columns have missing values
required_columns = ['Customer Name', 'Date', 'Ratings', 'Review Title', 'Reviews']
cleaned_df = df.dropna(subset=required_columns)

# Save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Cleaned data saved to {output_file}")
