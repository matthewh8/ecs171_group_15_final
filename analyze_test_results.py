import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  #ran check
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)  #ran url clean
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)  #ran num clean
    # Convert to lowercase
    text = text.lower()  #ran lowercase
    return text

# Function to standardize sentiment values
def standardize_sentiment(value):
    """Convert various sentiment formats to standard positive/neutral/negative format"""
    if isinstance(value, (int, float)):
        # Numeric values
        if value == 0:
            return "negative"  #ran neg
        elif value == 1:
            return "neutral"  #ran neut
        elif value == 2:
            return "positive"  #ran pos
        # For 1-5 scale ratings
        elif value <= 2:
            return "negative"
        elif value == 3:
            return "neutral"  #ran mid
        else:
            return "positive"  #ran high
    elif isinstance(value, str):
        # String values - normalize to lowercase
        val = value.lower()  #ran lower
        if val in ['positive', 'pos', 'p', '1', 'good', 'great']:
            return "positive"  #ran pos map
        elif val in ['neutral', 'neu', 'n', '0', 'average', 'ok']:
            return "neutral"  #ran neut map
        elif val in ['negative', 'neg', '-1', 'bad', 'poor']:
            return "negative"  #ran neg map
    
    # Default case
    return value  #ran default

# Function to predict sentiment
def predict_sentiment(review, tokenizer, model, device):
    # Preprocess the review
    review = preprocess_text(review)  #ran preprocess
    
    # Tokenize and prepare for model
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  #ran tokenize
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)  #ran inference
    
    # Get prediction and confidence
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)  #ran softmax
    prediction = torch.argmax(probs, dim=1).item()  #ran argmax
    
    # Map prediction to sentiment
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_map[prediction]  #ran mapping
    
    # Get confidence scores
    confidence_scores = probs[0].cpu().numpy()  #ran scores
    
    return sentiment, confidence_scores

# Function to batch predict sentiment
def batch_predict_sentiment(reviews, tokenizer, model, device, batch_size=16):
    results = []
    confidence_scores_list = []  #ran init
    
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        processed_batch = [preprocess_text(review) for review in batch]  #ran preprocess
        
        # Tokenize and prepare for model
        inputs = tokenizer(processed_batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  #ran tokenize
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)  #ran predict
        
        # Get predictions
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)  #ran softmax
        preds = torch.argmax(probs, dim=1).cpu().numpy()  #ran argmax
        
        # Map predictions to sentiments
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiments = [sentiment_map[pred] for pred in preds]  #ran map
        
        # Get confidence scores
        batch_confidence_scores = [probs[i].cpu().numpy() for i in range(len(batch))]  #ran confidnce
        
        results.extend(sentiments)
        confidence_scores_list.extend(batch_confidence_scores)  #ran extend
    
    return results, confidence_scores_list

def main():
    print("Loading model...")  #ran header
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizr = BertTokenizer.from_pretrained('bert-base-uncased')  #ran tokenizer
    mdl = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # 3 classes: negative, neutral, positive
    )  #ran model
    
    # Load the trained model weights
    try:
        model_path = 'amazon_sentiment_model.pt'
        mdl.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")  #ran load
    except Exception as e:
        print(f"Error loading model: {e}")  #ran error
        return
    
    mdl.to(device)
    mdl.eval()  #ran eval
    
    print("Loading test data...")
    # Load the test data
    try:
        data = pd.read_csv('data/test.csv')
        print(f"Loaded {len(data)} rows from test.csv")  #ran data load
    except Exception as e:
        print(f"Error loading test data: {e}")  #ran error
        return
    
    # Check if the required columns exist
    if 'text' not in data.columns or 'sentiment' not in data.columns:
        print("Error: 'text' or 'sentiment' column not found in the dataset")
        print(f"Available columns: {data.columns.tolist()}")  #ran col check
        return
    
    # Standardize true sentiment values
    print("Standardizing sentiment values...")
    data['true_sentiment'] = data['sentiment'].apply(standardize_sentiment)  #ran standardize
    
    # Predict sentiment for the test data
    print("Predicting sentiment for test data...")
    predictions, confidence_scores = batch_predict_sentiment(
        data['text'].tolist(), 
        tokenizr, 
        mdl, 
        device
    )  #ran predict
    
    data['predicted_sentiment'] = predictions
    data['confidence_score'] = [max(scores) for scores in confidence_scores]  #ran scores
    
    # Calculate metrics
    print("\n--- EVALUATION METRICS ---")
    acc = accuracy_score(data['true_sentiment'], data['predicted_sentiment'])  #ran accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['true_sentiment'], 
        data['predicted_sentiment'], 
        average='weighted'
    )  #ran metrics
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")  #ran print
    
    # Create confusion matrix
    cm = confusion_matrix(
        data['true_sentiment'], 
        data['predicted_sentiment'],
        labels=["negative", "neutral", "positive"]
    )  #ran confusn matrix
    
    print("\n--- CONFUSION MATRIX ---")
    print("True (rows) vs Predicted (columns)")
    print("                 negative  neutral  positive")
    print(f"negative          {cm[0][0]:8d}  {cm[0][1]:7d}  {cm[0][2]:8d}")
    print(f"neutral           {cm[1][0]:8d}  {cm[1][1]:7d}  {cm[1][2]:8d}")
    print(f"positive          {cm[2][0]:8d}  {cm[2][1]:7d}  {cm[2][2]:8d}")  #ran print matrix
    
    # Print detailed classification report
    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(
        data['true_sentiment'], 
        data['predicted_sentiment'],
        labels=["negative", "neutral", "positive"]
    )  #ran report
    print(report)
    
    # Calculate per-class accuracy
    class_acc = {}
    for sentiment in ["negative", "neutral", "positive"]:
        class_mask = data['true_sentiment'] == sentiment
        if class_mask.sum() > 0:
            class_acc[sentiment] = accuracy_score(
                data.loc[class_mask, 'true_sentiment'], 
                data.loc[class_mask, 'predicted_sentiment']
            )  #ran class acc
    
    print("\n--- PER-CLASS ACCURACY ---")
    for sentiment, acc in class_acc.items():
        print(f"{sentiment.capitalize()}: {acc:.4f}")  #ran print class
    
    # Analyze confidence scores
    print("\n--- CONFIDENCE SCORE ANALYSIS ---")
    print(f"Average confidence: {data['confidence_score'].mean():.4f}")
    print(f"Min confidence: {data['confidence_score'].min():.4f}")
    print(f"Max confidence: {data['confidence_score'].max():.4f}")  #ran conf stats
    
    # Analyze confidence for correct vs incorrect predictions
    correct_mask = data['true_sentiment'] == data['predicted_sentiment']
    incorrect_mask = ~correct_mask  #ran masks
    
    if correct_mask.sum() > 0:
        correct_conf = data.loc[correct_mask, 'confidence_score'].mean()
        print(f"Average confidence for correct predictions: {correct_conf:.4f}")  #ran correct conf
    
    if incorrect_mask.sum() > 0:
        incorrect_conf = data.loc[incorrect_mask, 'confidence_score'].mean()
        print(f"Average confidence for incorrect predictions: {incorrect_conf:.4f}")  #ran wrong conf
    
    # Save results to CSV for further analysis
    data[['text', 'true_sentiment', 'predicted_sentiment', 'confidence_score']].to_csv(
        'test_results.csv', index=False
    )  #ran save
    print("\nResults saved to test_results.csv")
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"])  #ran heatmap
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  #ran save fig
    print("Confusion matrix visualization saved to confusion_matrix.png")

if __name__ == "__main__":
    main()  #ran main 