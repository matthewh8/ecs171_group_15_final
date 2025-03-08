import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(file_path):
    """Load and prepare data from CSV/TSV file"""
    # Load data
    print(f"Loading data from {file_path}...")
    
    # Determine file format based on extension
    if file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    
    # Display dataset information
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Map column names based on what's available in the dataset
    if 'Review' in df.columns:
        review_col = 'Review'
    elif 'review' in df.columns:
        review_col = 'review'
    else:
        review_col = df.columns[0]  # Assume first column is review text
    
    if 'Liked' in df.columns:
        sentiment_col = 'Liked'
    elif 'sentiment' in df.columns:
        sentiment_col = 'sentiment'
    elif 'sentiment_label' in df.columns:
        sentiment_col = 'sentiment_label'
    else:
        sentiment_col = df.columns[1] if len(df.columns) > 1 else None
    
    print(f"Using {review_col} as review column and {sentiment_col} as sentiment column")
    
    # Preprocess text
    df['review_clean'] = df[review_col].apply(preprocess_text)
    
    # Prepare sentiment labels
    if sentiment_col:
        # Check if sentiment is already text labels or binary/numerical values
        if df[sentiment_col].dtype == 'object' and set(df[sentiment_col].unique()).issubset({'positive', 'neutral', 'negative'}):
            # Already text labels
            df['sentiment'] = df[sentiment_col]
            sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        else:
            # Binary/numerical labels, assume 1=positive, 0=negative
            label_map = {1: 'positive', 0: 'negative'}
            df['sentiment'] = df[sentiment_col].map(label_map)
            sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    else:
        # No sentiment column found, cannot proceed
        raise ValueError("No sentiment column found in the dataset")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['sentiment'].value_counts())
    
    # Create train-test split
    reviews = df['review_clean'].values
    targets = df['sentiment_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, sentiment_mapping

def train_model(model, train_data_loader, optimizer, device, epochs=4):
    """Train the BERT model"""
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(train_data_loader):
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
            
            # Update weights
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            
            _, predictions = torch.max(logits, dim=1)
            correct_predictions += torch.sum(predictions == targets)
            total_predictions += targets.shape[0]
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = correct_predictions.double() / total_predictions
        
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def evaluate_model(model, test_data_loader, device):
    """Evaluate the trained model"""
    model.eval()
    
    predictions = []
    actual_labels = []
    review_texts = []
    
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
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
    report = classification_report(actual_labels, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    return predictions, actual_labels, review_texts

def visualize_results(predictions, actual_labels, review_texts, sentiment_mapping):
    """Create visualizations for model results"""
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'review': review_texts,
        'actual': actual_labels,
        'predicted': predictions
    })
    
    # Map numerical labels back to text labels
    reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
    results_df['actual_sentiment'] = results_df['actual'].map(reverse_mapping)
    results_df['predicted_sentiment'] = results_df['predicted'].map(reverse_mapping)
    
    # Add a column to check if prediction was correct
    results_df['correct'] = results_df['actual'] == results_df['predicted']
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(actual_labels, predictions)
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(reverse_mapping.values()),
        yticklabels=list(reverse_mapping.values())
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 2. Class Distribution Bar Chart
    fig = px.histogram(
        results_df, 
        x='actual_sentiment',
        color='predicted_sentiment',
        barmode='group',
        title='Actual vs Predicted Sentiment Distribution',
        labels={'actual_sentiment': 'Actual Sentiment', 'count': 'Count', 'predicted_sentiment': 'Predicted Sentiment'}
    )
    fig.write_html('sentiment_distribution.html')
    
    # 3. Sample correct and incorrect predictions
    incorrect_preds = results_df[~results_df['correct']].sample(min(10, sum(~results_df['correct'])))
    
    # Display incorrect predictions with actual and predicted sentiment
    print("\nSample of Incorrect Predictions:")
    for i, row in incorrect_preds.iterrows():
        print(f"Review: {row['review'][:100]}...")
        print(f"Actual: {row['actual_sentiment']}, Predicted: {row['predicted_sentiment']}\n")
    
    return results_df

def extract_insights(results_df, reviews_df):
    """Extract business insights from sentiment analysis results"""
    # Assuming reviews_df has additional columns like 'product_id', 'date', etc.
    # Merge results with original data to access additional information
    insights_df = pd.merge(
        results_df,
        reviews_df[['review', 'product_id', 'date']], 
        left_on='review', 
        right_on='review',
        how='left'
    )
    
    # 1. Product sentiment analysis
    product_sentiment = insights_df.groupby('product_id')['predicted_sentiment'].value_counts().unstack().fillna(0)
    
    # Calculate sentiment score (-1 for negative, 0 for neutral, 1 for positive)
    product_sentiment['score'] = (
        product_sentiment['positive'] - product_sentiment['negative']
    ) / product_sentiment.sum(axis=1)
    
    # Sort by sentiment score
    product_sentiment = product_sentiment.sort_values('score', ascending=False)
    print(product_sentiment)
    
    # 2. Sentiment trends over time
    insights_df['date'] = pd.to_datetime(insights_df['date'])
    insights_df['month'] = insights_df['date'].dt.to_period('M')
    
    sentiment_over_time = (
        insights_df.groupby(['month', 'predicted_sentiment'])
        .size()
        .unstack()
        .fillna(0)
    )
    
    # Calculate sentiment ratio over time
    sentiment_over_time['ratio_positive'] = sentiment_over_time['positive'] / sentiment_over_time.sum(axis=1)
    
    # 3. Extract common phrases from each sentiment category
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Function to extract top n-grams
    def get_top_ngrams(corpus, n=2, top_k=10):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    
    # Get top phrases for each sentiment
    top_phrases = {}
    for sentiment in ['negative', 'neutral', 'positive']:
        corpus = insights_df[insights_df['predicted_sentiment'] == sentiment]['review'].values
        if corpus.size == 0:
            top_phrases[sentiment] = list()
            continue
        top_phrases[sentiment] = get_top_ngrams(corpus, n=2, top_k=10)
    
    # Print insights
    print("\n===== BUSINESS INSIGHTS =====")
    
    print("\nTop 5 Products by Sentiment Score:")
    print(product_sentiment[['positive', 'negative', 'score']].head(5))
    
    print("\nBottom 5 Products by Sentiment Score:")
    print(product_sentiment[['positive', 'negative', 'score']].tail(5))
    
    print("\nCommon Phrases in Positive Reviews:")
    for phrase, count in top_phrases['positive']:
        print(f"{phrase}: {count}")
    
    print("\nCommon Phrases in Negative Reviews:")
    for phrase, count in top_phrases['negative']:
        print(f"{phrase}: {count}")


    
    return product_sentiment, sentiment_over_time, top_phrases

def create_dashboard(results_df, product_sentiment, sentiment_over_time, top_phrases):
    """Create an HTML dashboard with key insights"""
    import plotly.figure_factory as ff
    
    # Format data for tables
    top_products = product_sentiment.head(5).reset_index()
    top_products_table = ff.create_table(top_products)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Review Sentiment Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .card {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">Customer Review Sentiment Analysis Dashboard</h1>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Overall Sentiment Distribution</div>
                        <div class="card-body">
                            <div id="sentiment-distribution"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">Sentiment Trends Over Time</div>
                        <div class="card-body">
                            <div id="sentiment-trends"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">Top Products by Sentiment</div>
                        <div class="card-body">
                            <div id="product-sentiment"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Common Phrases in Positive Reviews</div>
                        <div class="card-body">
                            <ul class="list-group">
                                {''.join([f'<li class="list-group-item d-flex justify-content-between align-items-center">{phrase}<span class="badge bg-primary rounded-pill">{count}</span></li>' for phrase, count in top_phrases['positive']])}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Common Phrases in Negative Reviews</div>
                        <div class="card-body">
                            <ul class="list-group">
                                {''.join([f'<li class="list-group-item d-flex justify-content-between align-items-center">{phrase}<span class="badge bg-danger rounded-pill">{count}</span></li>' for phrase, count in top_phrases['negative']])}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Load external charts
            var sentiment_dist = document.getElementById('sentiment-distribution');
            Plotly.newPlot(sentiment_dist, {{data: {{}}, layout: {{}}}});
            
            var sentiment_trends = document.getElementById('sentiment-trends');
            Plotly.newPlot(sentiment_trends, {{data: {{}}, layout: {{}}}});
            
            var product_sentiment = document.getElementById('product-sentiment');
            Plotly.newPlot(product_sentiment, {{data: {{}}, layout: {{}}}});
            
            // Load charts from files
            fetch('sentiment_distribution.html')
                .then(response => response.text())
                .then(html => {{
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const plotDiv = doc.querySelector('div[id^="plotly-"]');
                    if (plotDiv) {{
                        sentiment_dist.id = plotDiv.id;
                        sentiment_dist.className = plotDiv.className;
                        sentiment_dist.innerHTML = plotDiv.innerHTML;
                    }}
                }});
            
            fetch('sentiment_trends.html')
                .then(response => response.text())
                .then(html => {{
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const plotDiv = doc.querySelector('div[id^="plotly-"]');
                    if (plotDiv) {{
                        sentiment_trends.id = plotDiv.id;
                        sentiment_trends.className = plotDiv.className;
                        sentiment_trends.innerHTML = plotDiv.innerHTML;
                    }}
                }});
                
            fetch('product_sentiment.html')
                .then(response => response.text())
                .then(html => {{
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const plotDiv = doc.querySelector('div[id^="plotly-"]');
                    if (plotDiv) {{
                        product_sentiment.id = plotDiv.id;
                        product_sentiment.className = plotDiv.className;
                        product_sentiment.innerHTML = plotDiv.innerHTML;
                    }}
                }});
        </script>
    </body>
    </html>
    """
    
    # Save the dashboard HTML file
    with open('sentiment_dashboard.html', 'w') as f:
        f.write(html_content)
    
    print("\nDashboard created successfully: sentiment_dashboard.html")

def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for data directory
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Check for models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    
    # First try to use processed data from quicktest.py
    if os.path.exists('data/processed_reviews.csv'):
        data_file = 'data/processed_reviews.csv'
    # Then try the original TSV file
    elif os.path.exists('data/Restaurant_Reviews.tsv'):
        data_file = 'data/Restaurant_Reviews.tsv'
    # If neither exists, show error
    else:
        print("Error: No data file found. Please run quicktest.py first or create a data file.")
        print("Expected files: data/processed_reviews.csv or data/Restaurant_Reviews.tsv")
        return
    
    print(f"Using data file: {data_file}")
    X_train, X_test, y_train, y_test, sentiment_mapping = load_and_prepare_data(data_file)
    
    # 2. Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # 3 classes: negative, neutral, positive
    )
    model.to(device)
    
    # 3. Create datasets and data loaders
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)
    
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=16)
    
    # 4. Set up optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5,no_deprecation_warning=True)
    
    # 5. Train the model
    train_model(model, train_data_loader, optimizer, device, epochs=4)
    
    # 6. Evaluate the model
    predictions, actual_labels, review_texts = evaluate_model(model, test_data_loader, device)
    
    # 7. Visualize results
    results_df = visualize_results(predictions, actual_labels, review_texts, sentiment_mapping)
    
    # 8. Extract business insights
    # For demonstration, we'll use a dummy reviews_df - in a real scenario, you'd use your full dataset
    reviews_df = pd.DataFrame({
        'review': review_texts,
        'product_id': [f'PROD{i%10:03d}' for i in range(len(review_texts))],
        'date': pd.date_range(start='2023-01-01', periods=len(review_texts))
    })
    
    product_sentiment, sentiment_over_time, top_phrases = extract_insights(results_df, reviews_df)
    
    # 9. Create dashboard
    create_dashboard(results_df, product_sentiment, sentiment_over_time, top_phrases)
    
    # 10. Save the model
    model_path = 'models/sentiment_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

if __name__ == "__main__":
    main()