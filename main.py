from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from sentiment_analysis import SentimentAnalyzer
from trend_analysis import TrendAnalyzer
from dashboard import Dashboard

def main():
    """Main execution pipeline"""
    print("Starting Customer Review Intelligence Platform...")
    
    # Initialize components
    data_loader = DataLoader(sample_size=5000)  #reduced
    feature_engineer = FeatureEngineer()
    sentiment_analyzer = SentimentAnalyzer()
    trend_analyzer = TrendAnalyzer()
    
    
    print("\nStep 1: Data Loading")
    df = data_loader.load_data()
    df = data_loader.preprocess_data(df)
    
    
    print("\n Step 2: Feature Engineering ")
    feature_df, tfidf_features, topic_features = feature_engineer.create_feature_matrix(df)
    topics = feature_engineer.get_topic_words()
    
    
    print("\n Step 3: Sentiment Analysis ")
    sample_texts = df['review_text_clean'].head(100).tolist()  # Sample for demo
    sentiment_predictions = sentiment_analyzer.predict_sentiment(sample_texts)
    
    
    print("\n Step 4: Trend Analysis")
    insights = trend_analyzer.generate_insights(df, feature_df, topics)
    
    
    print("\n Step 5: Dashboard Creation")
    dashboard = Dashboard()
    dashboard.create_dashboard(df, feature_df, sentiment_predictions, topics, insights)
    
    print("\n Platform setup complete!")
    print("Run with: streamlit run your_script.py")

if __name__ == "__main__":
    main()