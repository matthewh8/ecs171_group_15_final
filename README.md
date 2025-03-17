# Amazon Review Sentiment Analysis

This project implements a BERT-based sentiment analysis model for Amazon product reviews, classifying them as negative, neutral, or positive.

## Features
- Fine-tuned BERT model for sentiment analysis
- Interactive Streamlit dashboard for real-time predictions
- Support for both GPU and CPU training
- Configurable training parameters (sample size, epochs, batch size)
- Built-in data preprocessing pipeline (included in train_amazon_model.py)
- Complete sentiment analysis functionality (included in both files)

## Requirements
```bash
pip install torch
pip install transformers
pip install pandas
pip install numpy
pip install scikit-learn
pip install streamlit
pip install tqdm
pip install seaborn
pip install matplotlib
```

## Project Structure
- `train_amazon_model.py`: Complete training pipeline including:
  - Data preprocessing (text cleaning, rating conversion)
  - Automatic column detection for reviews and ratings
  - Dataset splitting and preparation
  - Model training and evaluation
- `streamlit_dashboard.py`: Interactive web interface including:
  - Real-time sentiment prediction
  - Confidence score visualization
  - Built-in text preprocessing
  - No additional preprocessing files needed

## Training the Model
1. Place your Amazon reviews dataset (CSV format) in a `data` folder
2. Run the training script:
```bash
python train_amazon_model.py --sample_size 50000 --epochs 4 --batch_size 32
```

Parameters:
- `--sample_size`: Number of reviews to use for training (default: 50000)
- `--epochs`: Number of training epochs (default: 4)
- `--batch_size`: Batch size for training (default: 32)

## Using the Dashboard
1. Ensure the trained model file (`amazon_sentiment_model.pt`) is in the project root
2. Launch the Streamlit dashboard:
```bash
streamlit run streamlit_dashboard.py
```

## Model Performance
- Training accuracy: 96.42%
- Test accuracy: 88.07%
- Supports three sentiment classes: negative, neutral, positive

## Notes
- The model uses BERT (bert-base-uncased) as the base architecture
- Training requires significant computational resources; GPU is recommended
- All necessary preprocessing and analysis functions are contained within the two main Python files
- No additional utility files are needed for preprocessing or sentiment analysis

## License
MIT License 