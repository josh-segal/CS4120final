import os
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    precision = precision_score(dev_y, preds)
    recall = recall_score(dev_y, preds)
    accuracy = accuracy_score(dev_y, preds)
    f1 = f1_score(dev_y, preds)
    # print("f1:", f1)

    return precision, recall, f1, accuracy


def create_presentation_to_paper_mapping(presentation_folder, paper_folder):
    presentation_to_paper = {}
    presentation_files = sorted(os.listdir(presentation_folder))
    paper_files = sorted(os.listdir(paper_folder))

    # Assuming presentation and paper files have the same indexes
    for idx, (presentation_file, paper_file) in enumerate(zip(presentation_files, paper_files)):
        presentation_index = presentation_file.split('.')[0]  # Extract index from presentation filename
        paper_index = paper_file.split('.')[0]  # Extract index from paper filename
        assert presentation_index == paper_index, "Mismatch between presentation and paper indexes"
        presentation_to_paper[presentation_index] = idx  # Store index instead of paper name

    return presentation_to_paper


def process_papers_folder(folder_path):
    """
    Process files in a folder using a given parse function.

    Args:
    - folder_path (str): Path to the folder containing files.
    - parse_function (function): Function to parse the content of each file.

    Returns:
    - all_data (list): List of processed data from all files in the folder.
    """
    all_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # Read the file content
            file_content = read_file(file_path)
            if file_content is not None:
                # Parse the content using the provided parse function
                parsed_data = parse_paper_xml(file_content)
                if parsed_data:
                    all_data.extend(parsed_data)
    return all_data


def process_presentation_folder(folder_path):
    """
    Process files in a folder using a given parse function.

    Args:
    - folder_path (str): Path to the folder containing files.
    - parse_function (function): Function to parse the content of each file.

    Returns:
    - all_data (list): List of processed data from all files in the folder.
    """
    all_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # Read the file content
            file_content = read_file(file_path)
            if file_content is not None:
                # Parse the content using the provided parse function
                parsed_data = parse_presentation_xml(file_content)
                if parsed_data:
                    all_data.extend(parsed_data)
    return all_data
