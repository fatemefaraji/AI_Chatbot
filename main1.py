import os
import json
import random
import logging
import argparse

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Initialize NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None, config=None):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings or {}
        self.config = config or {
            'batch_size': 8,
            'learning_rate': 0.001,
            'epochs': 100,
            'validation_split': 0.2,
            'patience': 10  # For early stopping
        }

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        """Tokenize and lemmatize input text."""
        if not text or not isinstance(text, str):
            return []
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text.lower())
        return [lemmatizer.lemmatize(word) for word in words if word.isalnum()]

    def bag_of_words(self, words):
        """Create a bag-of-words representation."""
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        """Parse intents from JSON file."""
        self.vocabulary = []
        self.documents = []
        self.intents = []
        self.intents_responses = {}

        try:
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading intents: {e}")
            raise ValueError(f"Failed to load intents: {e}")

        if not intents_data.get('intents'):
            logger.error("Intents JSON is empty or malformed")
            raise ValueError("Intents JSON is empty or malformed")

        for intent in intents_data['intents']:
            tag = intent.get('tag')
            patterns = intent.get('patterns', [])
            responses = intent.get('responses', [])
            if not tag or not patterns:
                logger.warning(f"Skipping invalid intent: {tag}")
                continue

            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = responses

            for pattern in patterns:
                if not pattern.strip():
                    continue
                pattern_words = self.tokenize_and_lemmatize(pattern)
                if pattern_words:
                    self.documents.append((pattern_words, tag))
                    self.vocabulary.extend(pattern_words)

        self.vocabulary = sorted(set(self.vocabulary))
        if not self.vocabulary:
            logger.error("Vocabulary is empty")
            raise ValueError("No valid patterns found in intents")

        logger.info(f"Parsed {len(self.intents)} intents and {len(self.vocabulary)} unique words")

    def prepare_data(self):
        """Prepare training data."""
        bags = []
        indices = []

        for document in self.documents:
            words, intent = document
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(intent)
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        logger.info(f"Prepared data: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def train_model(self):
        """Train the chatbot model with validation and early stopping."""
        if self.X is None or self.y is None:
            logger.error("Data not prepared. Call prepare_data() first")
            raise ValueError("Data not prepared")

        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Split into train and validation
        val_size = int(self.config['validation_split'] * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            train_loss = 0.0
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader) if val_loader else 1
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth', weights_only=True))
        logger.info("Training completed")

    def save_model(self, model_path, dimensions_path):
        """Save model and dimensions."""
        try:
            torch.save(self.model.state_dict(), model_path)
            with open(dimensions_path, 'w') as f:
                json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path, dimensions_path):
        """Load model and dimensions."""
        try:
            with open(dimensions_path, 'r') as f:
                dimensions = json.load(f)
            self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def process_message(self, input_message):
        """Process user message and return response."""
        if not input_message or not isinstance(input_message, str) or not input_message.strip():
            logger.warning("Invalid or empty input message")
            return "Please provide a valid message."

        words = self.tokenize_and_lemmatize(input_message)
        if not words:
            logger.warning("No valid words after tokenization")
            return "I didn't understand that. Try again?"

        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        logger.debug(f"Predicted intent: {predicted_intent}")

        # Handle function mappings
        if predicted_intent in self.function_mappings:
            try:
                response = self.function_mappings[predicted_intent]()
                if response:
                    return response
            except Exception as e:
                logger.error(f"Error executing function for intent {predicted_intent}: {e}")

        # Fallback to intent responses
        responses = self.intents_responses.get(predicted_intent, [])
        return random.choice(responses) if responses else "I'm not sure how to respond to that."

def get_stocks():
    """Return a string with random stock recommendations."""
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']
    selected = random.sample(stocks, 3)
    return f"Recommended stocks: {', '.join(selected)}"

def main():
    parser = argparse.ArgumentParser(description="Chatbot Assistant")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--intents', default='intents.json', help="Path to intents JSON file")
    parser.add_argument('--model', default='chatbot_model.pth', help="Path to model file")
    parser.add_argument('--dimensions', default='dimensions.json', help="Path to dimensions file")
    args = parser.parse_args()

    assistant = ChatbotAssistant(args.intents, function_mappings={'stocks': get_stocks})

    if args.train:
        logger.info("Starting training mode")
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model()
        assistant.save_model(args.model, args.dimensions)
    else:
        logger.info("Starting interactive mode")
        assistant.parse_intents()
        assistant.prepare_data()  # Needed for vocabulary
        assistant.load_model(args.model, args.dimensions)

        while True:
            message = input("Enter your message (or /quit to exit): ").strip()
            if message.lower() == '/quit':
                logger.info("Exiting chatbot")
                break
            response = assistant.process_message(message)
            print(response)

if __name__ == '__main__':
    main()
