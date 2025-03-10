# Amazon Sentiment Analysis Training Script for Google Colab
# Copy and paste this entire script into a new Google Colab notebook

# Step 1: Install required packages
!pip install transformers pandas numpy matplotlib seaborn scikit-learn tqdm

# Step 2: Import libraries
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
import time
import re
import os
from google.colab import files

# Step 3: Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("WARNING: GPU is not enabled. Go to Runtime > Change runtime type and select GPU.")

# Step 4: Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Step 5: Create directories
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('models'):
    os.makedirs('models')

# Step 6: Upload the Amazon dataset
print("Please upload the Amazon dataset (amazon_unlocked_mobile.csv)")
uploaded = files.upload()

for filename in uploaded.keys():
    if 'amazon' in filename.lower() and '.csv' in filename.lower():
        with open(f'data/{filename}', 'wb') as f:
            f.write(uploaded[filename])
        data_file = f'data/{filename}'
        print(f"Saved dataset to {data_file}")

# Step 7: Define the ReviewDataset class
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

# Step 8: Define text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 9: Load and prepare the data
def prepare_data(file_path, sample_size=50000):
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
    
    return X_train, X_test, y_train, y_test, sentiment_mapping, review_column

# Step 10: Define training function
def train_model(model, train_loader, optimizer, scheduler, device, epochs=4):
    # Track training time
    start_time = time.time()
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
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
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            running_loss += loss.item()
            
            _, predictions = torch.max(logits, dim=1)
            correct += torch.sum(predictions == targets).item()
            total += targets.shape[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTraining complete! Total time: {hours}h {minutes}m {seconds}s")
    
    return model

# Step 11: Define evaluation function
def evaluate_model(model, test_loader, device, sentiment_mapping):
    model.eval()
    
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
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
    
    return accuracy

# Step 12: Load and prepare the data
try:
    data_file = 'data/' + list(uploaded.keys())[0]  # Use the uploaded file
    # You can adjust the sample size here (default: 50000)
    sample_size = 50000  # Increase this for better results, decrease if you run out of memory
    X_train, X_test, y_train, y_test, sentiment_mapping, review_column = prepare_data(data_file, sample_size)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise e

# Step 13: Set up the model, optimizer, and scheduler
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # 3 classes: negative, neutral, positive
)
model.to(device)

# Create datasets and data loaders
train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

batch_size = 32 if torch.cuda.is_available() else 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
total_steps = len(train_loader) * 4  # 4 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Step 14: Train the model
model = train_model(model, train_loader, optimizer, scheduler, device, epochs=4)

# Step 15: Evaluate the model
accuracy = evaluate_model(model, test_loader, device, sentiment_mapping)

# Step 16: Save and download the model
torch.save(model.state_dict(), 'amazon_sentiment_model.pt')
print("Model saved as 'amazon_sentiment_model.pt'")

# Download the model and results
files.download('amazon_sentiment_model.pt')
files.download('amazon_confusion_matrix.png')

print("Files downloaded. Please upload amazon_sentiment_model.pt to your project directory.") 