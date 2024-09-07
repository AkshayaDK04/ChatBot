"""import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
f = open("C:/Users/Smile/Documents/Chatbot_Project/data.txt", 'r', errors="ignore")
raw_doc = f.read().lower()

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Tokenization and Lemmatization
sentencetoks = nltk.sent_tokenize(raw_doc)
wordtoks = nltk.word_tokenize(raw_doc)
lemmer = nltk.stem.WordNetLemmatizer()

def lemtoks(toks):
    return [lemmer.lemmatize(tok) for tok in toks]

removepunctuation = dict((ord(punct), None) for punct in string.punctuation)

def lemnormalise(text):
    return lemtoks(nltk.word_tokenize(text.lower().translate(removepunctuation)))

# Greeting Function
greet_ip = ('hello', 'hi', 'whatsup', 'how are you?', 'shall we talk')
greet_op = ('Hi!', 'Hello!', 'Hey there! How are you doing?', 'Hello! I hope you are good.')

def greet(input):
    for word in input.split():
        if word.lower() in greet_ip:
            return random.choice(greet_op)
    return None

# Response Function
def response(userinput):
    nlp_resp = ''
    sentencetoks.append(userinput)
    TfidfV = TfidfVectorizer(tokenizer=lemnormalise, stop_words='english')
    tfidf = TfidfV.fit_transform(sentencetoks)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf < 0.1:  # Use a threshold to check if similarity is meaningful
        nlp_resp = "I am sorry, I don't understand."
    else:
        nlp_resp = sentencetoks[idx]
    sentencetoks.remove(userinput)
    return nlp_resp

# Chat Flow
flag = True
print("Hello! How can I help you today?")

while(flag):
    userinp = input().lower()
    if 'bye' in userinp:
        flag = False
        print("Goodbye!")
    elif userinp == 'thank you' or userinp == 'thanks':
        flag = False
        print('You are welcome!')
    else:
        if greet(userinp) is not None:
            print(greet(userinp))
        else:
            print(response(userinp))
"""

# Import necessary libraries
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys




# Download necessary NLTK data packages
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load and preprocess the data
def load_corpus(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            raw_doc = f.read()
            return raw_doc
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit()

corpus = load_corpus("C:/Users/Smile/Documents/Chatbot_Project/data.txt")

# Tokenization
sentence_tokens = nltk.sent_tokenize(corpus)  # Converts corpus to a list of sentences
word_tokens = nltk.word_tokenize(corpus.lower())  # Converts corpus to a list of words in lowercase

# Word Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()



def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

TfidfV = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
tfidf_corpus = TfidfV.fit_transform(sentence_tokens)

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Hey!", "Hi!", "Greetings!", "Hello! How can I assist you today?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response to user queries
def response(userinput):
    nlp_resp = ''
    sentence_tokens.append(userinput)  # Add the user's input to the sentence tokens
    TfidfV = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfV.fit_transform(sentence_tokens)  # Compute the TF-IDF matrix
    
    # Compute cosine similarity
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # Find the second-highest similarity index
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]  # Second-highest value in the sorted similarity list
    
    # Ensure that the query itself is not returned as a response
    if req_tfidf == 0 or sentence_tokens[idx] == userinput:
        nlp_resp = "I am sorry, I don't understand."
    else:
        nlp_resp = sentence_tokens[idx]
    
    sentence_tokens.pop()  # Remove the user's input from the sentence tokens
    return nlp_resp


# Main function to start the chatbot
"""def chatbot():
    print("CollegeBot: Hello! I am here to answer your queries about the college. Type 'bye' to exit.")
    
    while True:
        user_response = input("User:").lower()
        if user_response == 'bye':
            print("CollegeBot: Goodbye! Have a nice day.")
            break
        elif user_response in ['thanks', 'thank you']:
            print("CollegeBot: You're welcome!")
            break
        elif greeting(user_response) is not None:
            print(f"CollegeBot: {greeting(user_response)}")
        else:
            print(f"CollegeBot: {response(user_response)}")

# Start the chatbot
if __name__ == "__main__":
    chatbot()"""



"""

import warnings
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Suppress warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Open the data file that contains information about the college
f = open("C:/Users/Smile/Documents/Chatbot_Project/data.txt", 'r', errors="ignore")
raw_doc = f.read().lower()  # Read and convert the text to lowercase

# Download necessary NLTK resources for tokenizing and lemmatization
nltk.download('punkt')  # Punkt tokenizer (for sentence and word tokenization)
nltk.download('wordnet')  # WordNet dictionary (for lemmatization)

# Tokenization - Break the raw document into sentences and words
sentencetoks = nltk.sent_tokenize(raw_doc)  # Sentence tokens (splits the document into sentences)
wordtoks = nltk.word_tokenize(raw_doc)  # Word tokens (splits the document into individual words)

# Lemmatizer - Convert words to their base forms
lemmer = nltk.stem.WordNetLemmatizer()

# Lemmatization function to normalize words
def lemtoks(toks):
    return [lemmer.lemmatize(tok) for tok in toks]  # Lemmatizes each token in the input list

# Remove punctuation from text (creates a dictionary to replace punctuation with None)
removepunctuation = dict((ord(punct), None) for punct in string.punctuation)

# Function to normalize text: tokenize, convert to lowercase, remove punctuation, and lemmatize
def lemnormalise(text):
    return lemtoks(nltk.word_tokenize(text.lower().translate(removepunctuation)))

# Greeting function: Matches user input with greeting keywords and returns a random response
greet_ip = ('hello', 'hi', 'whatsup', 'how are you?', 'shall we talk')
greet_op = ('Hi!', 'Hello!', 'Hey there! How are you doing?', 'Hello! I hope you are good.')

def greet(input):
    for word in input.split():  # Split input into words and check if any word matches the greeting keywords
        if word.lower() in greet_ip:
            return random.choice(greet_op)  # Return a random greeting response
    return None

# Keyword-based matching function for college-related questions
def keyword_match(query):
    query = query.lower()  # Convert the user's input to lowercase
    keywords = {
        'courses': 'The college offers undergraduate and postgraduate courses.',
        'admissions': 'Admissions are open from June to August.',
        'facilities': 'The campus has a library, sports complex, and hostel facilities.'
    }
    # Search for a keyword in the query
    for keyword, response in keywords.items():
        if re.search(keyword, query):
            return response  # Return the matching response
    return None

# Main response function that uses both keyword matching and TF-IDF similarity
def response(userinput):
    # Check if a keyword matches before using the NLP model
    keyword_resp = keyword_match(userinput)
    if keyword_resp:
        return keyword_resp  # If a keyword match is found, return the corresponding response
    
    # TF-IDF Vectorizer - Converts text into numerical values based on term frequency
    sentencetoks.append(userinput)  # Append the user's input to the list of sentence tokens
    TfidfV = TfidfVectorizer(tokenizer=lemnormalise, stop_words='english')  # Create TF-IDF vectorizer
    tfidf = TfidfV.fit_transform(sentencetoks)  # Transform sentence tokens into TF-IDF matrix

    # Calculate cosine similarity between the user input and all other sentences
    vals = cosine_similarity(tfidf[-1], tfidf)  # Compare the last sentence (user input) with the others
    idx = vals.argsort()[0][-2]  # Get the index of the sentence most similar to the user input
    flat = vals.flatten()  # Flatten the cosine similarity values
    flat.sort()  # Sort the similarity values
    req_tfidf = flat[-2]  # Get the second highest similarity score (the highest is the input itself)

    # If similarity is too low, return a default "I don't understand" response
    if req_tfidf < 0.1:
        nlp_resp = "I am sorry, I don't understand."
    else:
        nlp_resp = sentencetoks[idx]  # Return the most similar sentence from the original data

    sentencetoks.remove(userinput)  # Remove the user input from the sentence tokens
    return nlp_resp

# Chat flow - Continuously take user input until the user says 'bye'
flag = True
print("Hello! How can I help you today?")

while(flag):
    userinp = input().lower()  # Take input from the user and convert it to lowercase
    if 'bye' in userinp:  # End the chat if the user says 'bye'
        flag = False
        print("Goodbye!")
    elif userinp == 'thank you' or userinp == 'thanks':  # Handle polite endings
        flag = False
        print('You are welcome!')
    else:
        # Greet the user if the input matches a greeting; otherwise, use the response function
        if greet(userinp) is not None:
            print(greet(userinp))
        else:
            print(response(userinp))
"""