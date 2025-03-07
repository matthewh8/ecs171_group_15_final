import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_restaurant_reviews(file_path):
    """
    Load restaurant review data from TSV file
    
    Parameters:
    file_path (str): Path to the TSV file
    
    Returns:
    pd.DataFrame: Processed dataframe with reviews
    """
    # Read TSV file (tab-separated values)
    df = pd.read_csv(file_path, sep='\t')
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def preprocess_reviews(df, review_col='Review', sentiment_col='Liked'):
    """
    Preprocess the reviews dataset
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    review_col (str): Name of the column containing reviews
    sentiment_col (str): Name of the column containing sentiment labels
    
    Returns:
    pd.DataFrame: Processed dataframe with additional columns
    """
    # Copy the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Clean review text
    processed_df['review_clean'] = processed_df[review_col].apply(preprocess_text)
    
    # Map sentiment labels if needed
    # For the restaurant dataset, assuming: 
    # 1 = positive, 0 = negative (adapt as needed)
    sentiment_mapping = {0: 'negative', 1: 'positive'}
    
    # Create a text sentiment column
    if sentiment_col in processed_df.columns:
        processed_df['sentiment'] = processed_df[sentiment_col].map(sentiment_mapping)
        
        # Create numerical labels for model training
        # 0: negative, 1: neutral (if exists), 2: positive
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        processed_df['sentiment_label'] = processed_df['sentiment'].map(label_mapping)
    
    # Adding a date column if not present (using current date as placeholder)
    # In a real project, you would extract this from your data
    if 'date' not in processed_df.columns:
        processed_df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Create a product_id column if not present (using random assignment for demo)
    # In a real project, you would extract this from your data
    if 'product_id' not in processed_df.columns:
        processed_df['product_id'] = [f'REST{i%20:03d}' for i in range(len(processed_df))]
    
    return processed_df

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = df['review_clean'].values
    y = df['sentiment_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = load_restaurant_reviews('data/Restaurant_Reviews.tsv')
    
    # Preprocess reviews
    processed_df = preprocess_reviews(df)
    
    # Display processed data
    print("\nProcessed data sample:")
    print(processed_df.head())
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = split_dataset(processed_df)
    
    # Save processed data
    processed_df.to_csv('data/processed_reviews.csv', index=False)
    print("Processed data saved to 'data/processed_reviews.csv'")