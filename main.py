import nltk
import numpy as np
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Sample data
qa_pairs = {
    "Hi": "Hello! How can I assist you today?",
    "What is your name?": "I'm a chatbot created to assist you.",
    "How can I apply for the chatbot internship?": "You can apply by visiting our careers page and submitting your application.",
    "Thank you": "You're welcome! If you have any other questions, feel free to ask."
}

corpus = list(qa_pairs.keys())
responses = list(qa_pairs.values())

#text preprocessing


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#implement TF-IDF Vectorization and Response Generation

vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

def get_response(user_input):
    
    corpus_with_input = corpus + [user_input]
    
    tfidf = vectorizer.fit_transform(corpus_with_input)
    
    cosine_similarities = cosine_similarity(tfidf[-1],tfidf [:-1])
    
    idx = cosine_similarities.argsort()[0][-1]
    
    if cosine_similarities[0, idx] > 0.2:
        return responses[idx]
    
    else:
        return "I'm sorry, I don't understand. Could you please rephrase?"

while True:
    user_input = input("you: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print(f"chatbot: {response}")