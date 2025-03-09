# Restaurant Review Sentiment Analysis

A deep learning-based sentiment analysis system for restaurant reviews using BERT, with a streamlit dashboard for visualization.

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for faster training)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ecs171_group_15_final.git
cd ecs171_group_15_final
```

### 2. Install PyTorch

#### For Windows/Linux:
```bash
pip install torch torchvision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cuX.X
```

#### For Mac:
```bash
pip3 install torch torchvision
```

For specific configurations, visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to get the appropriate installation command for your system.

### 3. Install required packages

#### For Windows/Linux:
```bash
pip install pandas numpy matplotlib seaborn tqdm scikit-learn transformers plotly streamlit
```

#### For Mac:
```bash
pip3 install pandas numpy matplotlib seaborn tqdm scikit-learn transformers plotly streamlit
```

## Dataset

The system is designed to work with restaurant review data. The default dataset is expected to be in the following format:

- TSV file with columns for reviews and sentiment (1 for positive, 0 for negative)
- Place your data file in the `data/` directory as `Restaurant_Reviews.tsv`

## Usage

### 1. Quick Test (Recommended First Step)

Run the quick test script to verify all components are working correctly:

#### For Windows/Linux:
```bash
python quicktest.py
```

#### For Mac:
```bash
python3 quicktest.py
```

This script will:
- Check if the dataset is available
- Test data loading and preprocessing
- Verify model inference capabilities
- Check if all dashboard dependencies are available

### 2. Train the Sentiment Analysis Model

#### For Windows/Linux:
```bash
python sentiment_analyzer.py
```

#### For Mac:
```bash
python3 sentiment_analyzer.py
```

This script will:
- Load and preprocess the restaurant review data
- Train a BERT-based model for sentiment classification
- Evaluate the model's performance
- Generate visualizations of the results
- Save the trained model to `models/sentiment_model.pt`
- Create a static HTML dashboard

### 3. Launch the Interactive Dashboard

#### For Windows/Linux:
```bash
streamlit run streamlit_dashboard.py
```

#### For Mac:
```bash
streamlit run streamlit_dashboard.py
```
(The streamlit command is the same on all platforms after installation)

The dashboard provides:
- Data upload and processing
- Sentiment analysis for individual reviews
- Visualizations of overall sentiment distribution
- Product performance analysis
- Sentiment trends over time
- Common phrases in positive and negative reviews
- Sample reviews for each sentiment category

## Project Structure

- `dataloader.py`: Functions for loading and preprocessing review data
- `quicktest.py`: Quick test script to verify the setup
- `sentiment_analyzer.py`: Main script for training and evaluating the model
- `streamlit_dashboard.py`: Interactive dashboard for sentiment analysis
- `data/`: Directory for dataset files
- `models/`: Directory for saved model weights

## Example Workflow

1. Place your dataset in the `data/` directory
2. Run `python quicktest.py` to verify setup
3. Run `python sentiment_analyzer.py` to train the model
4. Run `streamlit run streamlit_dashboard.py` to launch the dashboard
5. Use the dashboard to analyze new reviews or uploaded datasets

## Notes

- The model uses BERT, which requires significant computational resources
- For large datasets, consider using a smaller sample size for training
- GPU acceleration is highly recommended for model training
