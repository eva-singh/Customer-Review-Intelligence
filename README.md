Customer Review Intelligence
This project is an attempt to create a Customer Review Intelligence Platform that helps us analyze large volumes of customer reviews to extract valuable insights.
It uses Sentiment Analysis, Topic Modeling, and Trend Analysis to summarize customer opinions, detect trends, and visualize actionable insights through an interactive dashboard.

-Key Features Implemented:
1.Data Pipeline
2.Advanced NLP Processing
  2.1 Semantic Analysis
  2.2 Topic Modelling
  2.3 Feature Engineering
  2.4 Trend Detection
3.ML Models
Pretrained models like distilBERT
4.Interactive Dashboard

-Dataset Source
This project uses the "McAuley-Lab/Amazon-Reviews-2023" dataset from Hugging Face.
The dataset was downloaded using the datasets library and stored locally as a CSV file for efficient loading and processing.
No need to manually download the dataset.
link: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

-How It Works?
I have created separate .py notebooks for different implementations that are independently reusable and scalable for future changes etc. The following define how each one functions:
main.py: connects everything and is responsible for workflow
data_loader.py: dataset loading 
sentiment_analysis.py: applies sentiment analysis
rend_analysis.py: generates trend insights
dashboard.py: launches the interactive dashboard 

-Tech Stack:
Python 
Natural Language Processing (NLP)
Streamlit
Hugging Face datasets
Pandas
Scikit Learn
PyTorch
Matplolib


