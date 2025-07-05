import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class DataLoader:
    
    
    def __init__(self, dataset_name="McAuley-Lab/Amazon-Reviews-2023", subset="All_Beauty", sample_size=10000):
        self.dataset_name = dataset_name
        self.subset = subset
        self.sample_size = sample_size
        
    def load_data(self):
        """Load and sample the dataset"""
        print(f"Loading {self.dataset_name} dataset...")
        
        try:
            # Try loading from Hugging Face
            dataset = load_dataset(self.dataset_name, self.subset, split="train")
            df = pd.DataFrame(dataset)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to sample dataset generation...")
            # Create sample data for demonstration
            df = self.create_sample_data()
        
        # Sample the dataset
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
        
        # Standardize column names
        column_mapping = {
            'text': 'review_text',
            'review_body': 'review_text',
            'review': 'review_text',
            'stars': 'rating',
            'star': 'rating',
            'score': 'rating',
            'product_category': 'category',
            'category': 'category'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['review_text', 'rating']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        print(f"Loaded {len(df)} reviews")
        return df
    
    def create_sample_data(self):
        """Create sample review data for demonstration"""
        print("Creating sample review data...")
        
        sample_reviews = [
            ("This product is amazing! Best purchase ever. Highly recommend.", 5),
            ("Decent quality for the price. Nothing special but works well.", 3),
            ("Terrible experience. Product broke after one week. Waste of money.", 1),
            ("Good value for money. Fast shipping and great customer service.", 4),
            ("Not as described. Quality is poor and doesn't match the photos.", 2),
            ("Excellent quality and fast delivery. Will buy again!", 5),
            ("Average product. Works as expected but nothing outstanding.", 3),
            ("Poor quality materials. Disappointed with this purchase.", 2),
            ("Outstanding product! Exceeded my expectations completely.", 5),
            ("Mediocre at best. There are better alternatives available.", 3),
        ]
        
        # Replicate sample data to create larger dataset
        extended_reviews = []
        for i in range(self.sample_size):
            review, rating = sample_reviews[i % len(sample_reviews)]
            extended_reviews.append({
                'review_text': review,
                'rating': rating,
                'asin': f'B00{i:06d}',
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
            })
        
        return pd.DataFrame(extended_reviews)
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        print("Preprocessing data...")
        
        # Clean text
        df['review_text_clean'] = df['review_text'].apply(self.clean_text)
        
        # Remove empty reviews
        df = df[df['review_text_clean'].str.len() > 10]
        
        # Add text features
        df['review_length'] = df['review_text_clean'].str.len()
        df['word_count'] = df['review_text_clean'].str.split().str.len()
        df['readability_score'] = df['review_text_clean'].apply(
            lambda x: flesch_reading_ease(x) if x else 0
        )
        
        # Convert rating to sentiment categories
        df['sentiment_label'] = df['rating'].apply(self.rating_to_sentiment)
        
        print(f"Preprocessed {len(df)} reviews")
        return df
    
    def rating_to_sentiment(self, rating):
        """Convert rating to sentiment label"""
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'