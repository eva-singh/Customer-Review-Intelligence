import streamlit as st
import plotly.express as px
from wordcloud import WordCloud

import pandas as pd
import numpy as np # numpy will be installed as a dependency of pandas/spacy
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

class Dashboard:
    """Creates interactive dashboard using Streamlit"""
    
    def __init__(self):
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Customer Review Intelligence Platform",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def create_dashboard(self, df, feature_df, sentiment_predictions, topics, insights):
        """Create the main dashboard"""
        st.title("🔍 Customer Review Intelligence Platform")
        st.markdown("---")
        
        #sidebar 
        st.sidebar.header("Filters")
        
        #sentiment
        sentiment_filter = st.sidebar.multiselect(
            "Select Sentiment",
            options=df['sentiment_label'].unique(),
            default=df['sentiment_label'].unique()
        )
        
        #rating
        rating_filter = st.sidebar.slider(
            "Rating Range",
            min_value=int(df['rating'].min()),
            max_value=int(df['rating'].max()),
            value=(int(df['rating'].min()), int(df['rating'].max()))
        )
        
        
        filtered_df = df[
            (df['sentiment_label'].isin(sentiment_filter)) &
            (df['rating'] >= rating_filter[0]) &
            (df['rating'] <= rating_filter[1])
        ]
        
        #main 
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(filtered_df))
        
        with col2:
            st.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
        
        with col3:
            positive_pct = (filtered_df['sentiment_label'] == 'positive').mean() * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        
        with col4:
            avg_length = filtered_df['review_length'].mean()
            st.metric("Avg Review Length", f"{avg_length:.0f} chars")
        
        #chart
        col1, col2 = st.columns(2)
        
        with col1:
            #sent dist
            fig_sentiment = px.pie(
                filtered_df,
                names='sentiment_label',
                title='Sentiment Distribution'
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            #rate dist
            fig_rating = px.histogram(
                filtered_df,
                x='rating',
                title='Rating Distribution'
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        #word cloud
        st.subheader("Word Cloud")
        if len(filtered_df) > 0:
            text = ' '.join(filtered_df['review_text_clean'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        
        #topics
        st.subheader("Discovered Topics")
        for i, topic in enumerate(topics):
            st.write(f"**{topic}**")
        
        
        st.subheader("Key Insights")
        for insight in insights:
            st.write(f"• {insight}")
        
        #
        st.subheader("Sample Reviews")
        sample_reviews = filtered_df.sample(min(5, len(filtered_df)))
        for idx, row in sample_reviews.iterrows():
            with st.expander(f"Rating: {row['rating']} | Sentiment: {row['sentiment_label']}"):
                st.write(row['review_text'])