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

class FeatureEngineer:
    """Handles feature engineering for text data"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=10
        )
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return tfidf_features, feature_names
    
    def extract_topics(self, tfidf_features):
        """Extract topics using LDA"""
        print("Extracting topics...")
        topic_features = self.lda_model.fit_transform(tfidf_features)
        return topic_features
    
    def get_topic_words(self, n_words=10):
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
            topics.append(f"Topic {topic_idx}: {' '.join(top_words)}")
        return topics
    
    def extract_entities(self, texts, sample_size=1000):
        """Extract named entities from text"""
        print("Extracting named entities...")
        entities = []
        
        
        sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
        
        for text in sample_texts:
            doc = self.nlp(text)
            text_entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities.extend(text_entities)
        
        return entities
    
    def create_feature_matrix(self, df):
        
        print("Creating feature matrix...")
        
        #tfidf
        tfidf_features, feature_names = self.extract_tfidf_features(df['review_text_clean'])
        
        
        topic_features = self.extract_topics(tfidf_features)
        
        #combine
        feature_df = pd.DataFrame({
            'review_length': df['review_length'],
            'word_count': df['word_count'],
            'readability_score': df['readability_score'],
            'rating': df['rating']
        })
        
        
        topic_df = pd.DataFrame(
            topic_features,
            columns=[f'topic_{i}' for i in range(topic_features.shape[1])]
        )
        
        feature_df = pd.concat([feature_df, topic_df], axis=1)
        
        return feature_df, tfidf_features, topic_features