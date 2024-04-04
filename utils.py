import os
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import shutil


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
        preprocessed_text.append(words)
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


def move_xml_files(source_dir, first_xml_dir, second_xml_dir):
    # Create destination directories if they don’t exist
    os.makedirs(first_xml_dir, exist_ok=True)
    os.makedirs(second_xml_dir, exist_ok=True)

    # Counter to handle duplicate file names
    file_counter = {}

    # Iterate through each folder in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        # Check if the item in the source directory is a folder
        if os.path.isdir(folder_path):
            # List all files in the current folder
            files = os.listdir(folder_path)
            # Filter out XML files
            xml_files = [file for file in files if file.endswith('.xml')]
            # Make sure there are exactly two XML files in the folder
            if len(xml_files) == 2:
                # Move the first XML file to the first_xml_dir
                first = xml_files[0]
                second = xml_files[1]
                if first.startswith('slide'):
                    first, second = second, first  # Swap if the first file is named 'slide'
                first_xml_file = os.path.join(folder_path, first)
                # Determine destination file name with counter
                first_destination = os.path.join(first_xml_dir, f"{folder_name}_{first}")
                # Increment counter if file name already exists
                file_counter[first_destination] = file_counter.get(first_destination, 0) + 1
                if file_counter[first_destination] > 1:
                    first_destination = f"{first_destination}_{file_counter[first_destination]}"
                shutil.move(first_xml_file, first_destination)
                # Move the second XML file to the second_xml_dir
                second_xml_file = os.path.join(folder_path, second)
                # Determine destination file name with counter
                second_destination = os.path.join(second_xml_dir, f"{folder_name}_{second}")
                # Increment counter if file name already exists
                file_counter[second_destination] = file_counter.get(second_destination, 0) + 1
                if file_counter[second_destination] > 1:
                    second_destination = f"{second_destination}_{file_counter[second_destination]}"
                shutil.move(second_xml_file, second_destination)
                print(f"Moved XML files from {folder_name} folder.")
    print("Process completed.")

