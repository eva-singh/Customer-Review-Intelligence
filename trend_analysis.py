import pandas as pd
import numpy as np # numpy will be installed as a dependency of pandas/spacy(issue faced before, thus commenting)
from datasets import load_dataset
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print("All libraries loaded successfully!")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrendAnalyzer:
    """Analyzes trends in review data"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        
    def sentiment_trends_over_time(self, df):
        """Analyze sentiment trends over time"""
        if 'date' not in df.columns:
            #test
            df['date'] = pd.date_range(
                start='2023-01-01',
                periods=len(df),
                freq='D'
            )
        
        #date group by
        trend_data = df.groupby([df['date'].dt.date, 'sentiment_label']).size().reset_index(name='count')
        
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Over Time', 'Rating Distribution', 
                          'Review Length vs Rating', 'Topic Distribution'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        #sentiment with time
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = trend_data[trend_data['sentiment_label'] == sentiment]
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['date'],
                    y=sentiment_data['count'],
                    name=sentiment,
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        #rating dist
        rating_counts = df['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=rating_counts.index, y=rating_counts.values, name='Rating Count'),
            row=1, col=2
        )
        
        # review vs length
        fig.add_trace(
            go.Scatter(
                x=df['rating'],
                y=df['review_length'],
                mode='markers',
                name='Length vs Rating',
                opacity=0.6
            ),
            row=2, col=1
        )
        
        #sent dist
        sentiment_counts = df['sentiment_label'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name='Sentiment Distribution'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Customer Review Analytics Dashboard")
        return fig
    
    def detect_anomalies(self, df):
        """Detect anomalies in review patterns"""
        
        df_sorted = df.sort_values('date') if 'date' in df.columns else df
        
        
        df_sorted['sentiment_score'] = df_sorted['sentiment_label'].map(
            {'negative': -1, 'neutral': 0, 'positive': 1}
        )
        
        df_sorted['rolling_sentiment'] = df_sorted['sentiment_score'].rolling(
            window=7, min_periods=1
        ).mean()
        
        
        df_sorted['sentiment_zscore'] = np.abs(
            (df_sorted['rolling_sentiment'] - df_sorted['rolling_sentiment'].mean()) / 
            df_sorted['rolling_sentiment'].std()
        )
        
        anomalies = df_sorted[df_sorted['sentiment_zscore'] > 2]
        
        return anomalies
    
    def generate_insights(self, df, feature_df, topics):
        """Generate actionable insights"""
        insights = []
        
        
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True)
        insights.append(f"Sentiment Distribution: {sentiment_dist.to_dict()}")
        
        #avg
        avg_rating = df['rating'].mean()
        insights.append(f"Average Rating: {avg_rating:.2f}/5")
        
        
        avg_length = df['review_length'].mean()
        insights.append(f"Average Review Length: {avg_length:.0f} characters")
        
        
        insights.append("Top Topics:")
        for topic in topics[:5]:
            insights.append(f"  - {topic}")
        
        #corr
        corr_rating_length = df['rating'].corr(df['review_length'])
        insights.append(f"Rating-Length Correlation: {corr_rating_length:.3f}")
        
        return insights
