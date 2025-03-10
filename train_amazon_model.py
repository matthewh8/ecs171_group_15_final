import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import argparse

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except:
    IN_COLAB = False
    print("Not running in Google Colab")

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=128):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and extra whitespace
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_amazon_data(file_path, sample_size=None):
    """Load and prepare the Amazon dataset for training"""
    print(f"Loading data from {file_path}...")
    
    # Load the dataset
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Original dataset size: {len(df)} reviews")
    
    # Sample the dataset if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} reviews from the dataset")
    
    # Find the review column
    review_column = None
    for col in ['Reviews', 'Review', 'review', 'reviews', 'Text', 'text', 'comment', 'Comment']:
        if col in df.columns:
            review_column = col
            break
    
    if review_column is None:
        raise ValueError("Could not find a review column in the dataset")
    print(f"Using '{review_column}' as the review column")
    
    # Find the rating column
    rating_column = None
    for col in ['Rating', 'Ratings', 'rating', 'ratings', 'Stars', 'stars', 'score', 'Score']:
        if col in df.columns:
            rating_column = col
            break
    
    if rating_column is None:
        raise ValueError("Could not find a rating column in the dataset")
    print(f"Using '{rating_column}' as the rating column")
    
    # Clean the data
    df = df.dropna(subset=[review_column, rating_column])
    print(f"After removing missing values: {len(df)} reviews")
    
    # Preprocess reviews
    print("Preprocessing reviews...")
    df['review_clean'] = df[review_column].apply(preprocess_text)
    
    # Convert ratings to sentiment labels (0: negative, 1: neutral, 2: positive)
    df['sentiment_label'] = df[rating_column].apply(
        lambda x: 0 if float(x) <= 2 else (1 if float(x) == 3 else 2)
    )
    
    # Create sentiment text labels for reference
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment'] = df['sentiment_label'].map(sentiment_mapping)
    
    # Print sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    print("\nSentiment distribution in the dataset:")
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} reviews ({count/len(df):.2%})")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_clean'].values, 
        df['sentiment_label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_label']
    )
    
    print(f"\nTraining set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")
    
    # Print class distribution
    train_class_dist = pd.Series(y_train).value_counts(normalize=True)
    test_class_dist = pd.Series(y_test).value_counts(normalize=True)
    
    print("\nClass distribution in training set:")
    for label, percentage in train_class_dist.items():
        print(f"{sentiment_mapping[label]}: {percentage:.2%}")
    
    print("\nClass distribution in test set:")
    for label, percentage in test_class_dist.items():
        print(f"{sentiment_mapping[label]}: {percentage:.2%}")
    
    return X_train, X_test, y_train, y_test, sentiment_mapping

def train_model(model, train_data_loader, optimizer, scheduler, device, epochs=4):
    """Train the BERT model"""
    # Track training time
    start_time = time.time()
    
    # Store training stats
    training_stats = []
    
    # Total training steps
    total_steps = len(train_data_loader) * epochs
    
    print(f"Starting training on {device}...")
    print(f"Total training steps: {total_steps}")
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start = time.time()
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar for batches
        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate metrics
            running_loss += loss.item()
            
            _, predictions = torch.max(logits, dim=1)
            correct_predictions += torch.sum(predictions == targets).item()
            total_predictions += targets.shape[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = correct_predictions / total_predictions
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_time:.2f}s")
        
        # Store stats
        training_stats.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'time': epoch_time
        })
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTraining complete! Total time: {hours}h {minutes}m {seconds}s")
    
    # Plot training stats
    stats_df = pd.DataFrame(training_stats)
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(stats_df['epoch'], stats_df['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(stats_df['epoch'], stats_df['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('amazon_training_stats.png')
    print("Saved training stats plot to 'amazon_training_stats.png'")
    
    return model

def evaluate_model(model, test_data_loader, device, sentiment_mapping):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    model.eval()
    
    predictions = []
    actual_labels = []
    review_texts = []
    
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
            review_texts.extend(batch['review_text'])
    
    # Calculate performance metrics
    accuracy = accuracy_score(actual_labels, predictions)
    conf_matrix = confusion_matrix(actual_labels, predictions)
    report = classification_report(actual_labels, predictions, target_names=list(sentiment_mapping.values()))
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(sentiment_mapping.values()),
                yticklabels=list(sentiment_mapping.values()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('amazon_confusion_matrix.png')
    print("Saved confusion matrix plot to 'amazon_confusion_matrix.png'")
    
    # Save some example predictions
    examples = []
    for i in range(min(10, len(review_texts))):
        examples.append({
            'review': review_texts[i],
            'actual': sentiment_mapping[actual_labels[i]],
            'predicted': sentiment_mapping[predictions[i]]
        })
    
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv('amazon_prediction_examples.csv', index=False)
    print("Saved prediction examples to 'amazon_prediction_examples.csv'")
    
    return predictions, actual_labels, review_texts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model on Amazon reviews')
    parser.add_argument('--sample_size', type=int, default=50000, help='Number of reviews to use for training (default: 50000)')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    args = parser.parse_args()
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # If in Colab, check for GPU
    if IN_COLAB and not torch.cuda.is_available():
        print("WARNING: GPU is not enabled in Colab. Go to Runtime > Change runtime type and select GPU.")
    
    # Check for data directory
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Check for models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    
    # If in Colab, download the dataset if needed
    if IN_COLAB:
        from google.colab import files
        import io
        
        # Check if dataset exists
        data_file = 'data/amazon_unlocked_mobile.csv'
        if not os.path.exists(data_file):
            print("Dataset not found. Please upload the Amazon dataset.")
            uploaded = files.upload()
            
            for filename in uploaded.keys():
                if 'amazon' in filename.lower() and '.csv' in filename.lower():
                    with open(f'data/{filename}', 'wb') as f:
                        f.write(uploaded[filename])
                    data_file = f'data/{filename}'
                    print(f"Saved dataset to {data_file}")
                    break
    else:
        # Load and prepare data
        data_file = 'data/amazon_unlocked_mobile.csv'
        if not os.path.exists(data_file):
            print(f"Error: Amazon dataset not found at {data_file}")
            return
    
    # Use a larger sample for training (adjust based on available memory)
    sample_size = args.sample_size  # Use command line argument
    print(f"Using sample size: {sample_size}")
    X_train, X_test, y_train, y_test, sentiment_mapping = load_and_prepare_amazon_data(data_file, sample_size)
    
    # Load BERT tokenizer and model
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # 3 classes: negative, neutral, positive
    )
    model.to(device)
    
    # Create datasets and data loaders
    print("Creating data loaders...")
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)
    
    # Use a larger batch size for faster training if GPU is available
    batch_size = args.batch_size if torch.cuda.is_available() else 16
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set up optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    
    # Set up learning rate scheduler
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train the model
    model = train_model(model, train_data_loader, optimizer, scheduler, device, epochs=args.epochs)
    
    # Evaluate the model
    evaluate_model(model, test_data_loader, device, sentiment_mapping)
    
    # Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), 'models/amazon_sentiment_model.pt')
    # Also save a copy in the root directory for the dashboard
    torch.save(model.state_dict(), 'amazon_sentiment_model.pt')
    print("Model saved: models/amazon_sentiment_model.pt and amazon_sentiment_model.pt")
    
    # If in Colab, download the model
    if IN_COLAB:
        files.download('amazon_sentiment_model.pt')
        files.download('amazon_confusion_matrix.png')
        files.download('amazon_training_stats.png')
        files.download('amazon_prediction_examples.csv')
        print("Files downloaded. Please upload amazon_sentiment_model.pt to your project directory.")

if __name__ == "__main__":
    main() 