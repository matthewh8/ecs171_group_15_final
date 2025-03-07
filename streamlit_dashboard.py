import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# Set page configuration
st.set_page_config(
    page_title="Customer Review Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # 3 classes: negative, neutral, positive
    )
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load('sentiment_model.pt', map_location=device))
        model.to(device)
        model.eval()
    except:
        st.warning("Trained model not found. Using base model.")
    return tokenizer, model, device

# Preprocess text
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment for a single review
def predict_sentiment(review, tokenizer, model, device):
    # Preprocess the review
    review = preprocess_text(review)
    
    # Tokenize the review
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move tensors to the correct device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    # Map prediction to sentiment
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_mapping[prediction.item()]
    
    # Get confidence scores using softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    confidence_scores = {
        'negative': probabilities[0].item(),
        'neutral': probabilities[1].item(),
        'positive': probabilities[2].item()
    }
    
    return sentiment, confidence_scores

# Batch predict sentiment for multiple reviews
def batch_predict_sentiment(reviews, tokenizer, model, device, batch_size=16):
    sentiments = []
    confidence_scores_list = []
    
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i+batch_size]
        processed_reviews = [preprocess_text(review) for review in batch_reviews]
        
        encodings = tokenizer.batch_encode_plus(
            processed_reviews,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs.logits, dim=1)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        batch_sentiments = [sentiment_mapping[pred.item()] for pred in predictions]
        
        batch_confidence_scores = []
        for prob in probabilities:
            confidence_scores = {
                'negative': prob[0].item(),
                'neutral': prob[1].item(),
                'positive': prob[2].item()
            }
            batch_confidence_scores.append(confidence_scores)
        
        sentiments.extend(batch_sentiments)
        confidence_scores_list.extend(batch_confidence_scores)
    
    return sentiments, confidence_scores_list

# Load the dataset
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    return None

# Extract insights from analyzed reviews
def extract_insights(df):
    # Ensure we have the right columns
    if 'predicted_sentiment' not in df.columns:
        return None, None, None
    
    # 1. Product sentiment analysis (if product_id exists)
    if 'product_id' in df.columns:
        product_sentiment = df.groupby('product_id')['predicted_sentiment'].value_counts().unstack().fillna(0)
        
        # Calculate sentiment score
        product_sentiment['score'] = (
            product_sentiment['positive'] - product_sentiment['negative']
        ) / product_sentiment.sum(axis=1)
        
        # Sort by sentiment score
        product_sentiment = product_sentiment.sort_values('score', ascending=False)
    else:
        product_sentiment = None
    
    # 2. Sentiment trends over time (if date exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        sentiment_over_time = (
            df.groupby(['month', 'predicted_sentiment'])
            .size()
            .unstack()
            .fillna(0)
        )
        
        # Calculate sentiment ratio over time
        sentiment_over_time['ratio_positive'] = sentiment_over_time['positive'] / sentiment_over_time.sum(axis=1)
    else:
        sentiment_over_time = None
    
    # 3. Extract common phrases from each sentiment category
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Function to extract top n-grams
    def get_top_ngrams(corpus, n=2, top_k=10):
        if len(corpus) == 0:
            return []
        
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    
    # Get top phrases for each sentiment
    top_phrases = {}
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in df['predicted_sentiment'].values:
            corpus = df[df['predicted_sentiment'] == sentiment]['review'].values
            top_phrases[sentiment] = get_top_ngrams(corpus, n=2, top_k=10)
        else:
            top_phrases[sentiment] = []
    
    return product_sentiment, sentiment_over_time, top_phrases

# Create streamlit app
def main():
    # Load model
    tokenizer, model, device = load_model()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Analyze Single Review", "Dashboard"])
    
    if page == "Upload Data":
        st.title("Customer Review Sentiment Analysis")
        st.header("Upload Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file with customer reviews", type="csv")
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.success(f"File uploaded successfully! {df.shape[0]} rows loaded.")
                
                # Display dataset information
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Column selection
                st.subheader("Column Selection")
                
                review_col = st.selectbox(
                    "Select the column containing reviews",
                    df.columns
                )
                
                optional_cols = st.multiselect(
                    "Select optional columns (product_id, date, etc.)",
                    [col for col in df.columns if col != review_col]
                )
                
                with st.expander("Data Processing Options"):
                    sample_size = st.slider("Sample size (for large datasets)", 
                                           min_value=100, 
                                           max_value=min(10000, df.shape[0]), 
                                           value=min(1000, df.shape[0]))
                    
                    handle_missing = st.checkbox("Drop rows with missing values", value=True)
                
                # Process button
                if st.button("Process Reviews"):
                    with st.spinner("Processing reviews..."):
                        # Sample data if needed
                        if sample_size < df.shape[0]:
                            analysis_df = df.sample(sample_size, random_state=42)
                            st.info(f"Using a sample of {sample_size} reviews")
                        else:
                            analysis_df = df.copy()
                        
                        # Handle missing values
                        if handle_missing:
                            pre_count = analysis_df.shape[0]
                            analysis_df = analysis_df.dropna(subset=[review_col])
                            post_count = analysis_df.shape[0]
                            if pre_count > post_count:
                                st.info(f"Dropped {pre_count - post_count} rows with missing reviews")
                        
                        # Extract reviews
                        reviews = analysis_df[review_col].astype(str).tolist()
                        
                        # Predict sentiments
                        sentiments, confidence_scores = batch_predict_sentiment(
                            reviews, tokenizer, model, device
                        )
                        
                        # Add predictions to dataframe
                        analysis_df['predicted_sentiment'] = sentiments
                        
                        # Add confidence scores
                        analysis_df['confidence_negative'] = [scores['negative'] for scores in confidence_scores]
                        analysis_df['confidence_neutral'] = [scores['neutral'] for scores in confidence_scores]
                        analysis_df['confidence_positive'] = [scores['positive'] for scores in confidence_scores]
                        
                        # Save the processed data to session state
                        st.session_state['analyzed_df'] = analysis_df
                        
                        # Display success message and navigate to dashboard
                        st.success("Reviews processed successfully! Navigate to Dashboard to view results.")
                
                # Check if we have analyzed data in session state
                if 'analyzed_df' in st.session_state:
                    st.subheader("Processed Data Preview")
                    st.dataframe(st.session_state['analyzed_df'].head())
                    
                    # Option to download processed data
                    csv = st.session_state['analyzed_df'].to_csv(index=False)
                    st.download_button(
                        label="Download processed data as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
        
    elif page == "Analyze Single Review":
        st.title("Analyze a Single Review")
        
        review_text = st.text_area("Enter a customer review", height=150)
        
        if st.button("Analyze Sentiment"):
            if review_text:
                with st.spinner("Analyzing sentiment..."):
                    sentiment, confidence_scores = predict_sentiment(
                        review_text, tokenizer, model, device
                    )
                    
                    # Display the result with appropriate color
                    if sentiment == 'positive':
                        st.success(f"Sentiment: {sentiment.upper()}")
                    elif sentiment == 'negative':
                        st.error(f"Sentiment: {sentiment.upper()}")
                    else:
                        st.info(f"Sentiment: {sentiment.upper()}")
                    
                    # Display confidence scores
                    st.subheader("Confidence Scores")
                    
                    # Create bar chart
                    fig = px.bar(
                        x=['Negative', 'Neutral', 'Positive'],
                        y=[confidence_scores['negative'], 
                           confidence_scores['neutral'], 
                           confidence_scores['positive']],
                        color=['Negative', 'Neutral', 'Positive'],
                        color_discrete_map={
                            'Negative': 'rgb(239, 85, 59)',
                            'Neutral': 'rgb(99, 110, 250)',
                            'Positive': 'rgb(72, 199, 142)'
                        }
                    )
                    fig.update_layout(
                        title="Sentiment Confidence Scores",
                        xaxis_title="Sentiment",
                        yaxis_title="Confidence",
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig)
                    
                    # Word highlighting (simplified version)
                    st.subheader("Review Text Analysis")
                    words = review_text.split()
                    
                    # This is a simplified approach - a more sophisticated approach would use attention weights
                    positive_words = ["good", "great", "excellent", "amazing", "love", "best", "perfect", "happy"]
                    negative_words = ["bad", "terrible", "worst", "awful", "poor", "disappointed", "hate", "horrible"]
                    
                    html_text = ""
                    for word in words:
                        clean_word = ''.join(c for c in word.lower() if c.isalnum())
                        if clean_word in positive_words:
                            html_text += f' <span style="background-color: rgba(72, 199, 142, 0.3);">{word}</span>'
                        elif clean_word in negative_words:
                            html_text += f' <span style="background-color: rgba(239, 85, 59, 0.3);">{word}</span>'
                        else:
                            html_text += f' {word}'
                    
                    st.markdown(html_text, unsafe_allow_html=True)
                    
                    # Add explanation
                    st.markdown("""
                    <small>Note: Words highlighted in <span style="background-color: rgba(72, 199, 142, 0.3);">green</span> are typically positive. 
                    Words highlighted in <span style="background-color: rgba(239, 85, 59, 0.3);">red</span> are typically negative.
                    This is a simple rule-based highlighting and does not reflect the model's actual attention.</small>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a review to analyze.")
                
    elif page == "Dashboard":
        st.title("Sentiment Analysis Dashboard")
        
        # Check if we have analyzed data
        if 'analyzed_df' not in st.session_state:
            st.warning("Please upload and process data first.")
            st.info("Go to the 'Upload Data' page to get started.")
            return
        
        # Get the analyzed data
        df = st.session_state['analyzed_df']
        
        # Extract insights
        product_sentiment, sentiment_over_time, top_phrases = extract_insights(df)
        
        # Dashboard layout
        st.header("Overall Sentiment Distribution")
        
        # Create sentiment distribution pie chart
        sentiment_counts = df['predicted_sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'negative': 'rgb(239, 85, 59)',
                'neutral': 'rgb(99, 110, 250)',
                'positive': 'rgb(72, 199, 142)'
            },
            hole=0.4
        )
        fig.update_layout(
            title="Overall Sentiment Distribution",
            legend_title="Sentiment"
        )
        st.plotly_chart(fig)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_reviews = len(df)
            st.metric("Total Reviews", total_reviews)
            
        with col2:
            positive_percentage = round((sentiment_counts.get('positive', 0) / total_reviews) * 100, 1)
            st.metric("Positive Reviews", f"{positive_percentage}%")
            
        with col3:
            negative_percentage = round((sentiment_counts.get('negative', 0) / total_reviews) * 100, 1)
            st.metric("Negative Reviews", f"{negative_percentage}%")
        
        # Product sentiment analysis (if available)
        if product_sentiment is not None:
            st.header("Product Sentiment Analysis")
            
            top_n = st.slider("Number of products to display", min_value=5, max_value=20, value=10)
            
            # Top products by sentiment score
            st.subheader(f"Top {top_n} Products by Sentiment Score")
            top_products = product_sentiment.head(top_n)
            
            fig = px.bar(
                top_products.reset_index(),
                x='product_id',
                y=['positive', 'neutral', 'negative'],
                title=f'Top {top_n} Products by Sentiment Score',
                color_discrete_map={
                    'positive': 'rgb(72, 199, 142)',
                    'neutral': 'rgb(99, 110, 250)',
                    'negative': 'rgb(239, 85, 59)'
                },
                barmode='group'
            )
            st.plotly_chart(fig)
            
            # Bottom products by sentiment score
            st.subheader(f"Bottom {top_n} Products by Sentiment Score")
            bottom_products = product_sentiment.tail(top_n).iloc[::-1]  # Reverse to show worst first
            
            fig = px.bar(
                bottom_products.reset_index(),
                x='product_id',
                y=['positive', 'neutral', 'negative'],
                title=f'Bottom {top_n} Products by Sentiment Score',
                color_discrete_map={
                    'positive': 'rgb(72, 199, 142)',
                    'neutral': 'rgb(99, 110, 250)',
                    'negative': 'rgb(239, 85, 59)'
                },
                barmode='group'
            )
            st.plotly_chart(fig)
        
        # Sentiment over time (if available)
        if sentiment_over_time is not None:
            st.header("Sentiment Trends Over Time")
            
            # Convert Period to string for plotting
            sentiment_plot_data = sentiment_over_time.reset_index()
            sentiment_plot_data['month'] = sentiment_plot_data['month'].astype(str)
            
            fig = px.line(
                sentiment_plot_data,
                x='month',
                y=['positive', 'neutral', 'negative'],
                title='Sentiment Trends Over Time',
                color_discrete_map={
                    'positive': 'rgb(72, 199, 142)',
                    'neutral': 'rgb(99, 110, 250)',
                    'negative': 'rgb(239, 85, 59)'
                }
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Reviews",
                legend_title="Sentiment"
            )
            st.plotly_chart(fig)
            
            # Positive ratio trend
            if 'ratio_positive' in sentiment_plot_data.columns:
                fig = px.line(
                    sentiment_plot_data,
                    x='month',
                    y='ratio_positive',
                    title='Positive Review Ratio Over Time'
                )
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Ratio of Positive Reviews",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig)
        
        # Common phrases
        if top_phrases:
            st.header("Common Phrases by Sentiment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Common Phrases in Positive Reviews")
                if top_phrases['positive']:
                    positive_phrases_df = pd.DataFrame(
                        top_phrases['positive'], 
                        columns=['Phrase', 'Count']
                    )
                    fig = px.bar(
                        positive_phrases_df,
                        y='Phrase',
                        x='Count',
                        title='Top Phrases in Positive Reviews',
                        color_discrete_sequence=['rgb(72, 199, 142)'],
                        orientation='h'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                else:
                    st.info("No positive reviews to analyze")
            
            with col2:
                st.subheader("Common Phrases in Negative Reviews")
                if top_phrases['negative']:
                    negative_phrases_df = pd.DataFrame(
                        top_phrases['negative'], 
                        columns=['Phrase', 'Count']
                    )
                    fig = px.bar(
                        negative_phrases_df,
                        y='Phrase',
                        x='Count',
                        title='Top Phrases in Negative Reviews',
                        color_discrete_sequence=['rgb(239, 85, 59)'],
                        orientation='h'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                else:
                    st.info("No negative reviews to analyze")
        
        # Sample reviews
        st.header("Sample Reviews")
        
        tab1, tab2, tab3 = st.tabs(["Positive Reviews", "Neutral Reviews", "Negative Reviews"])
        
        with tab1:
            positive_samples = df[df['predicted_sentiment'] == 'positive'].sample(min(5, sum(df['predicted_sentiment'] == 'positive')))
            if not positive_samples.empty:
                for i, row in positive_samples.iterrows():
                    st.markdown(f"**Review**: {row[df.columns[0]]}")
                    st.markdown("---")
            else:
                st.info("No positive reviews found")
        
        with tab2:
            neutral_samples = df[df['predicted_sentiment'] == 'neutral'].sample(min(5, sum(df['predicted_sentiment'] == 'neutral')))
            if not neutral_samples.empty:
                for i, row in neutral_samples.iterrows():
                    st.markdown(f"**Review**: {row[df.columns[0]]}")
                    st.markdown("---")
            else:
                st.info("No neutral reviews found")
        
        with tab3:
            negative_samples = df[df['predicted_sentiment'] == 'negative'].sample(min(5, sum(df['predicted_sentiment'] == 'negative')))
            if not negative_samples.empty:
                for i, row in negative_samples.iterrows():
                    st.markdown(f"**Review**: {row[df.columns[0]]}")
                    st.markdown("---")
            else:
                st.info("No negative reviews found")

if __name__ == "__main__":
    main()