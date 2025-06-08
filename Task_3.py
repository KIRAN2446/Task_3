import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt', quiet=True)


qa_pairs = {
    "hello": "Hey! How can I assist you today?",
    "how are you": "I'm a chatbot, here to help you!",
    "what is your name": "I'm your friendly AI chatbot.",
    "tell me a joke": "Why did the computer show up at work late? It had a hard drive!",
    "thank you": "You're welcome! Happy to help.",
    "bye": "Goodbye! Have a great day!"
}

questions = list(qa_pairs.keys())
answers = list(qa_pairs.values())

vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def preprocess(text):
    
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    max_index = similarities.argmax()
    max_score = similarities[max_index]

    if max_score < 0.3:
        return "Sorry, I didn't quite understand. Could you please rephrase?"
    else:
        return answers[max_index]

def main():
    print("Chatbot: Hi! Ask me anything. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'bye':
            print("Chatbot:", qa_pairs["bye"])
            break
        print("Chatbot:", get_response(user_input))

if __name__ == "__main__":
    main()
