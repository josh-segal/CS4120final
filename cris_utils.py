from bs4 import BeautifulSoup
import numpy as np
import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer

def read_folder(directory_path, folder_path):
  # Get a list of all files in the folder
  file_list = os.listdir(folder_path)

  # add the directory path to the beginning of each file path
  files = []
  for file_path in file_list:
    file_path = directory_path + file_path
    files.append(file_path)
  return files




def read_file(file_path):
    """
    Read the content of a file and return it as a string.

    Args:
    - file_path (str): The path to the file.

    Returns:
    - file_content (str): The content of the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def parse_xml(file_xml):
    """
    Parse XML and extract text data

    Args:
    - file_xml (str): XML data of the file

    Returns:
    - file_data (list): List of text extracted from <p> elements
    """
    file_data = []
    soup = BeautifulSoup(file_xml, 'xml')
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        file_data.append(paragraph.get_text())
    return file_data


def preprocess_text(text_data):
    """
    Clean and preprocess text data

    Args:
    - text_data (list): List of text data to be preprocessed

    Returns:
    - preprocessed_text (list): List of preprocessed text data
    """
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for text in text_data:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenization
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        # Stemming
        words = [stemmer.stem(word) for word in words]
        # Join tokens back to text
        preprocessed_text.append(words)
    return preprocessed_text

def read_embeddings(filename: str, tokenizer: Tokenizer) -> (dict, dict):
    '''Loads and parses embeddings trained in earlier.
    Parameters:
        filename (str): path to file
        Tokenizer: tokenizer used to tokenize the data (needed to get the word to index mapping)
    Returns:
        (dict): mapping from word to its embedding vector
        (dict): mapping from index to its embedding vector
    '''
    # YOUR CODE HERE
    word_map = {}
    index_map = {}
    vector = []
    word = ""

    #open file
    file = open(filename)

    #reading the file into a string
    spooky_content = file.readlines()
    #tokenizer.fit_on_texts(spooky_content)
    #tokenizer.texts_to_sequences(spooky_content)

    #map word to its embedding vector
    #map index to its embedding vector

    del spooky_content[0]

    for line in spooky_content:

        tokens  = line.split(" ")
        word = tokens[0]
        vector = []
        del tokens[0]
        tokens[-1] = tokens[-1][:-2]

        for token in tokens:
            vector.append(token)

        try:
            index_map[tokenizer.word_index[word]] = vector
            word_map[word] = vector
        except KeyError:
            tokenizer.word_index[word] = max(tokenizer.word_index.values()) + 1
            word_map[word] = vector

    return word_map, index_map

def find_most_similar_paper(paper_embeddings, pres_embedding):
    
    # Calculate cosine similarity between each paper and the desired presentation
    similarities = []
    for paper_embedding in paper_embeddings:
        # Calculate cosine similarity
        similarity = cosine_similarity([pres_embedding], [paper_embedding])[0][0]
        similarities.append(similarity)
    
    # Get the index of the most similar paper
    most_similar_index = np.argmax(similarities)
    
    return most_similar_index


