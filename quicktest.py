"""
Quick test script to verify the sentiment analysis pipeline
"""
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from dataloader import load_restaurant_reviews, preprocess_reviews, preprocess_text

def test_data_loading():
    """Test data loading and preprocessing"""
    print("\n===== Testing Data Loading =====")
    
    # Check for data directory
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Check for dataset
    dataset_path = 'data/Restaurant_Reviews.tsv'
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
        print("Please place the restaurant reviews TSV file in the data directory")
        return False
    
    # Test loading dataset
    try:
        df = load_restaurant_reviews(dataset_path)
        print(f"Successfully loaded dataset with {len(df)} reviews")
        
        # Test preprocessing
        processed_df = preprocess_reviews(df)
        print(f"Successfully preprocessed dataset")
        
        # Save processed data
        processed_df.to_csv('data/processed_reviews.csv', index=False)
        print("Saved processed data to 'data/processed_reviews.csv'")
        
        return True
    except Exception as e:
        print(f"Error loading/processing dataset: {str(e)}")
        return False

def test_model_inference():
    """Test model loading and inference"""
    print("\n===== Testing Model Inference =====")
    
    # Check for models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    
    # Check if model exists
    model_path = 'models/sentiment_model.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Model will be created when you run sentiment_analyzer.py")
        
        # Test with base BERT model
        print("Testing with base BERT model instead...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3  # 3 classes: negative, neutral, positive
        )
        
        # Load trained weights if available
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded trained model from {model_path}")
        
        model.to(device)
        model.eval()
        
        # Test inference with sample reviews
        sample_reviews = [
            "The food was amazing and the service was excellent!",
            "The restaurant was okay, nothing special.",
            "Terrible experience. The food was cold and the staff was rude."
        ]
        
        print("\nTesting sentiment analysis on sample reviews:")
        for review in sample_reviews:
            # Preprocess the review
            processed_review = preprocess_text(review)
            
            # Tokenize
            inputs = tokenizer(
                processed_review,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            
            # Map prediction to sentiment
            sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            predicted_sentiment = sentiment_mapping[prediction]
            
            print(f"\nReview: {review}")
            print(f"Prediction: {predicted_sentiment}")
        
        return True
    except Exception as e:
        print(f"Error during model inference: {str(e)}")
        return False

def test_dashboard_dependencies():
    """Test if all dashboard dependencies are available"""
    print("\n===== Testing Dashboard Dependencies =====")
    
    try:
        import streamlit
        import plotly.express
        import matplotlib.pyplot
        import seaborn
        
        print("All dashboard dependencies are available")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Run all tests"""
    print("=== Restaurant Review Sentiment Analysis Quick Test ===")
    
    # Test data pipeline
    data_ok = test_data_loading()
    
    # Test model inference
    model_ok = test_model_inference()
    
    # Test dashboard dependencies
    dashboard_ok = test_dashboard_dependencies()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Data Pipeline: {'✓ OK' if data_ok else '✗ Issues found'}")
    print(f"Model Inference: {'✓ OK' if model_ok else '✗ Issues found'}")
    print(f"Dashboard Dependencies: {'✓ OK' if dashboard_ok else '✗ Issues found'}")
    
    if data_ok and model_ok and dashboard_ok:
        print("\nAll systems ready! You can now:")
        print("1. Run 'python sentiment_analyzer.py' to train the model")
        print("2. Run 'streamlit run streamlit_dashboard.py' to launch the dashboard")
    else:
        print("\nPlease fix the issues above before proceeding")

if __name__ == "__main__":
    main()