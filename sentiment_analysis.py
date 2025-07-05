import pandas as pd
import numpy as np
from datasets import load_dataset
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch # Make sure torch is imported

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline 


class SentimentAnalyzer:
    """Handles sentiment analysis using pre-trained and fine-tuned models"""

    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SentimentAnalyzer: Initializing model '{self.model_name}' on device: {self.device}")

        #load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            #load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                torch_dtype=torch.float32  # Crucial for preventing meta loading????
            ).to("cpu") 

            #move model to target
            self.model.to(self.device)

            print(f"SentimentAnalyzer: Model '{self.model_name}' successfully moved to {self.device}.")

        except Exception as e:
            print(f"SentimentAnalyzer: ERROR loading or moving model '{self.model_name}' to {self.device}. Error: {e}")
            print("This often indicates insufficient GPU VRAM or an incompatibility.")
            print("Consider trying a smaller model (e.g., 'finiteautomata/bertweet-base-sentiment-analysis') or ensuring PyTorch/CUDA are correctly installed for your GPU.")
            raise 

        #set the model to eval
        self.model.eval()

    def preprocess_for_bert(self, texts, labels=None, max_length=512):
        """Preprocess text for BERT-style models"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        
        encodings = {key: val.to(self.device) for key, val in encodings.items()}

        if labels is not None:
            # Convert string labels to integers
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            labels = [label_map[label] for label in labels]
            encodings['labels'] = torch.tensor(labels).to(self.device) # Also move labels to device

        return encodings

    def fine_tune_model(self, train_texts, train_labels, val_texts, val_labels):
        """Fine-tune the model on domain-specific data"""
        print("Fine-tuning model...")

        # Prepare datasets - ensure data is on CPU before creating Dataset objects
        # The Trainer will handle moving batches to the device during training
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        train_labels_int = [label_map[label] for label in train_labels]
        val_labels_int = [label_map[label] for label in val_labels]

        train_encodings['labels'] = torch.tensor(train_labels_int)
        val_encodings['labels'] = torch.tensor(val_labels_int)


        class ReviewDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                
                return {key: val[idx] for key, val in self.encodings.items()}

            def __len__(self):
                return len(self.encodings['input_ids'])

        train_dataset = ReviewDataset(train_encodings)
        val_dataset = ReviewDataset(val_encodings)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            #
            per_device_train_batch_size=8, # Reduced from 16, meta issue before???
            per_device_eval_batch_size=8,  # Reduced from 16
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            
        )

        #initialise 
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # train
        trainer.train()

        #save
        trainer.save_model('./fine_tuned_model')

        return trainer

    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        self.model.eval() 
        predictions = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs.max().item()

                label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                predictions.append({
                    'sentiment': label_map[predicted_class],
                    'confidence': confidence,
                    'probabilities': probs.cpu().numpy()[0]
                })

        return predictions