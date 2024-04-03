from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


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
        preprocessed_text.append(' '.join(words))
    return preprocessed_text

#%%
