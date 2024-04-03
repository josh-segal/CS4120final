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


def parse_paper_xml(file_xml):
    """
    Parse XML and extract text data from 'head' elements, concatenating subsequent 'p' elements.

    Args:
    - file_xml (str): XML data of the file

    Returns:
    - file_data (list): List of text extracted from 'head' elements, with subsequent 'p' elements concatenated
    """
    file_data = []
    soup = BeautifulSoup(file_xml, 'xml')
    heads = soup.find_all('head')
    for head in heads:
        head_text = ""
        p_elements = head.find_next_siblings('p')
        for p_element in p_elements:
            text = p_element.get_text()
            if text.strip():  # Check if the text is not empty or whitespace
                head_text += text
        if head_text:  # Append to file_data only if head_text is not empty
            file_data.append(head_text)
    return file_data


def parse_presentation_xml(presentation_xml):
    """
    Parse XML of presentation and extract text data from 'page' divs, concatenating subsequent 'p' elements.

    Args:
    - presentation_xml (str): XML data of the presentation

    Returns:
    - presentation_data (list): List of text extracted from 'page' divs, with subsequent 'p' elements concatenated
    """
    presentation_data = []
    soup = BeautifulSoup(presentation_xml, 'xml')
    pages = soup.find_all('div', class_='page')
    for page in pages:
        # Initialize page text
        page_text = ""
        p_elements = page.find_all('p')
        for p_element in p_elements:
            text = p_element.get_text().strip()  # Get the text and strip leading/trailing whitespace
            if text:  # Check if the text is not empty
                page_text += text + " " # Add space between paragraphs
        if page_text:  # Append to presentation_data only if page_text is not empty
            presentation_data.append(page_text.strip())  # Strip leading/trailing whitespace
    return presentation_data


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
