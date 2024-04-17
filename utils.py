import os
from bs4 import BeautifulSoup
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import re
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import joblib


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
    flat_data = []
    soup = BeautifulSoup(file_xml, 'xml')
    body = soup.find('body')
    if body:
        p_elements = body.find_all('p')
        for p_element in p_elements:
            text = p_element.get_text()
            if text.strip():  # Check if the text is not empty or whitespace
                sentences = sent_tokenize(text)
                file_data.append(sentences)
        for sublist in file_data:
            flat_data.extend(sublist)
    return flat_data


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
                page_text += text + " "  # Add space between paragraphs
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
        # Tokenization (split text into sentences)
        sentences = sent_tokenize(text)
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove stopwords and stemming for each sentence
        preprocessed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [clean_word(word) for word in words if clean_word(word) not in stop_words]
            words = [stemmer.stem(word) for word in words]
            preprocessed_sentences.append(' '.join(words))
        preprocessed_text.append(preprocessed_sentences)
    return preprocessed_text


def clean_word(word):
    # Remove punctuation and special characters
    word = re.sub(r'[^\w\s]', '', word)
    # Convert to lowercase
    word = word.lower()
    return word


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


def organize_xml_folders(source_folder, papers_folder, presentations_folder):
    # Ensure destination folders exist
    os.makedirs(papers_folder, exist_ok=True)
    os.makedirs(presentations_folder, exist_ok=True)

    # Iterate through each folder in the source folder
    for index, folder_name in enumerate(sorted(os.listdir(source_folder))):
        # Create the full path to the current folder
        folder_path = os.path.join(source_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Find the XML files inside the current folder
            for filename in os.listdir(folder_path):
                if filename.startswith("Paper") and filename.endswith(".xml"):
                    paper_xml = os.path.join(folder_path, filename)
                    shutil.move(paper_xml, os.path.join(papers_folder, f"{index}_paper.xml"))
                elif filename.startswith("slide") and filename.endswith(".xml"):
                    presentation_xml = os.path.join(folder_path, filename)
                    shutil.move(presentation_xml, os.path.join(presentations_folder, f"{index}_presentation.xml"))


def parse_title(xml_content):
    root = ET.fromstring(xml_content)
    # Define the namespace
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    # Find the title element using the namespace
    title = root.find(".//tei:titleStmt/tei:title[@level='a'][@type='main']", namespaces=ns)
    if title is not None and title.text is not None:
        return title.text.strip()
    else:
        return None


def clean_data(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    return text


def change_lower(text):
    text = text.lower()
    return text


stopwords_list = stopwords.words("english")


def remover(text):
    text_tokens = text.split(" ")
    final_list = [word for word in text_tokens if not word in stopwords_list]
    text = ' '.join(final_list)
    return text

# Define the directory containing the models
models_dir = 'models'

# Join the path
model_path = os.path.join(models_dir, 'LSTM_RNN_MODEL.h5')

model = load_model(model_path)

# Load the tokenizer object
with open(os.path.join(models_dir, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Load the vectorizer object
with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# Load the logistic regression model
log_reg_model = joblib.load(os.path.join(models_dir, 'best_logistic_regression_model.pkl'))


# Load the model
# transformer_model = load_model('distilbert_trained_model.h5', custom_objects={'TFDistilBertModel': TFDistilBertModel})



def predict_log_reg(paper_file, presentation_file):
    paper_data, presentation_data = combine_data(paper_file, presentation_file)

    presentation_flat = [sentence for sublist in presentation_data for sentence in sublist]
    paper_flat = [sentence for sublist in paper_data for sentence in sublist]
    presentation_sentence_pairs = []
    for sentence in presentation_flat:
        most_similar_sentence, similarity_score = find_most_similar_sentence(sentence, paper_flat)
        presentation_sentence_pairs.append([sentence, most_similar_sentence, similarity_score])
        presentation_sentence_pairs = sorted(presentation_sentence_pairs, key=lambda x: x[2], reverse=True)

    presentation_sentences_list = []
    paper_sentences_list = []

    presentation_sentences = [sentences[0] for sentences in presentation_sentence_pairs]
    paper_sentences = [sentences[1] for sentences in presentation_sentence_pairs]
    presentation_sentences_list.append(presentation_sentences)
    paper_sentences_list.append(paper_sentences)

    combined_sentences = [presentation + ' ' + paper for presentation, paper in zip(presentation_sentences_list[0], paper_sentences_list[0])]

    text_data = vectorizer.transform(combined_sentences)

    probabilities = log_reg_model.predict_proba(text_data)

    # entail_probs = [prob[0] for prob in probabilities]

    predicted_classes = np.argmax(probabilities, axis=1)

    # Step 3: Class Labels
    class_labels = ["Entailment", "Neutral", "Contradictory"]  # Replace with your actual class labels
    predicted_labels = [class_labels[idx] for idx in predicted_classes]

    # Count the occurrences of each label
    label_counts = {label: predicted_labels.count(label) for label in set(predicted_labels)}

    # Calculate the total count of all labels
    total_count = sum(label_counts.values())

    # Calculate the proportion of each label
    label_proportions = {label: count / total_count for label, count in label_counts.items()}

    # Define weights for each label
    label_weights = {
        'Contradictory': 0,
        'Neutral': 0.5,
        'Entailment': 1
    }

    # Calculate the weighted sum of counts for all labels
    weighted_sum = sum(label_counts[label] * label_weights[label] for label in label_counts)

    # Normalize the weighted sum to range from 0 to 1
    normalized_weighted_sum = weighted_sum / (total_count * max(label_weights.values()))
    transformed_score = np.exp(normalized_weighted_sum * 10) / np.exp(5)

    # Define the thresholds
    thresholds = {
        'A VERY BAD': 0.3,
        'A BAD': 0.50,
        'A GOOD': 0.70,
        'A GREAT': 0.85,
        'AN EXCELLENT': 1.0
    }

    # Determine the category based on the normalized weighted sum
    category = 'AN EXCELLENT'
    for label, threshold in thresholds.items():
        if transformed_score <= threshold:
            category = label
            break

    return category, probabilities, transformed_score


def predict_LSTM_RNN(paper_file, presentation_file):
    paper_data, presentation_data = combine_data(paper_file, presentation_file)

    presentation_paper_pairs = []
    presentation_flat = [sentence for sublist in presentation_data for sentence in sublist]
    paper_flat = [sentence for sublist in paper_data for sentence in sublist]
    presentation_sentence_pairs = []
    for sentence in presentation_flat:
        most_similar_sentence, similarity_score = find_most_similar_sentence(sentence, paper_flat)
        presentation_sentence_pairs.append([sentence, most_similar_sentence, similarity_score])
        presentation_sentence_pairs = sorted(presentation_sentence_pairs, key=lambda x: x[2], reverse=True)

    presentation_sentences_list = []
    paper_sentences_list = []

    presentation_sentences = [sentences[0] for sentences in presentation_sentence_pairs]
    paper_sentences = [sublist[1] for sublist in presentation_sentence_pairs]
    presentation_sentences_list.append(presentation_sentences)
    paper_sentences_list.append(paper_sentences)

    # Tokenize inference sentences using the loaded tokenizer
    inference_premise_sequences = tokenizer.texts_to_sequences(presentation_sentences_list[0])
    inference_hypothesis_sequences = tokenizer.texts_to_sequences(paper_sentences_list[0])

    # Pad the sequences to the same maximum sequence length
    inference_premise_sequences = pad_sequences(inference_premise_sequences, maxlen=45, padding='post')
    inference_hypothesis_sequences = pad_sequences(inference_hypothesis_sequences, maxlen=45, padding='post')

    probabilities = model.predict([inference_premise_sequences, inference_hypothesis_sequences])

    # entail_probs = [prob[0] for prob in probabilities]

    predicted_classes = np.argmax(probabilities, axis=1)

    # Step 3: Class Labels
    class_labels = ["Entailment", "Neutral", "Contradictory"]  # Replace with your actual class labels
    predicted_labels = [class_labels[idx] for idx in predicted_classes]

    # Count the occurrences of each label
    label_counts = {label: predicted_labels.count(label) for label in set(predicted_labels)}

    # Calculate the total count of all labels
    total_count = sum(label_counts.values())

    # Calculate the proportion of each label
    label_proportions = {label: count / total_count for label, count in label_counts.items()}

    # Define weights for each label
    label_weights = {
        'Contradictory': 0,
        'Neutral': 0.5,
        'Entailment': 1
    }

    # Calculate the weighted sum of counts for all labels
    weighted_sum = sum(label_counts[label] * label_weights[label] for label in label_counts)

    # Normalize the weighted sum to range from 0 to 1
    normalized_weighted_sum = weighted_sum / (total_count * max(label_weights.values()))
    transformed_score = np.exp(normalized_weighted_sum * 10) / np.exp(5)

    # Define the thresholds
    thresholds = {
        'A VERY BAD': 0.3,
        'A BAD': 0.50,
        'A GOOD': 0.70,
        'A GREAT': 0.85,
        'AN EXCELLENT': 1.0
    }

    # Determine the category based on the normalized weighted sum
    category = None
    for label, threshold in thresholds.items():
        if transformed_score <= threshold:
            category = label
            break

    return category, probabilities, transformed_score


def predict_transformer(paper_file, presentation_file):
    # Load tokenizer and model
    t_model = transformer_model
    t_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    t_model = DistilBertForSequenceClassification.from_pretrained(t_model)

    # Combine data
    paper_data, presentation_data = combine_data(paper_file, presentation_file)

    presentation_flat = [sentence for sublist in presentation_data for sentence in sublist]
    paper_flat = [sentence for sublist in paper_data for sentence in sublist]

    presentation_sentence_pairs = []
    for sentence in presentation_flat:
        most_similar_sentence, similarity_score = find_most_similar_sentence(sentence, paper_flat)
        presentation_sentence_pairs.append([sentence, most_similar_sentence, similarity_score])

    presentation_sentences_list = [sentences[0] for sentences in presentation_sentence_pairs]
    paper_sentences_list = [sublist[1] for sublist in presentation_sentence_pairs]

    # Tokenize and pad sequences
    input_dict = t_tokenizer(presentation_sentences_list, paper_sentences_list, padding=True, truncation=True,
                             return_tensors='pt')

    # Model inference
    with torch.no_grad():
        outputs = t_model(**input_dict)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()

    # Calculate transformed score
    transformed_score = probabilities[:, 0]  # Assuming the third class is 'Entailment'

    # Define thresholds and determine category
    thresholds = {
        'A VERY BAD': 0.3,
        'A BAD': 0.50,
        'A GOOD': 0.70,
        'A GREAT': 0.85,
        'AN EXCELLENT': 1.0
    }

    category = None
    for label, threshold in thresholds.items():
        if transformed_score <= threshold:
            category = label
            break

    return category, probabilities, transformed_score


def process_file(file_path, parse_func, preprocess_func):
    file_content = file_path.read()
    parsed_data = parse_func(file_content)
    preprocessed_data = preprocess_func(parsed_data)
    return preprocessed_data


def combine_data(paper_file, presentation_file):
    # paper_file = paper_file.read()
    # presentation_file = presentation_file.read()
    paper_data = process_file(paper_file, parse_paper_xml, preprocess_text)
    presentation_data = process_file(presentation_file, parse_presentation_xml, preprocess_text)

    # combined_data = {"paper": paper_data, "presentation": presentation_data}
    return paper_data, presentation_data


def find_most_similar_sentence(query_sentence, sentences):
    # Combine query sentence with the list of sentences
    all_sentences = [query_sentence] + sentences

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF vectors for all sentences
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    # Calculate cosine similarity between query sentence and all sentences
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Find the index of the most similar sentence
    most_similar_index = similarity_scores.argmax()

    # Return the most similar sentence and its similarity score
    most_similar_sentence = sentences[most_similar_index]
    similarity_score = similarity_scores[most_similar_index]

    return most_similar_sentence, similarity_score

papers_manually_matched: dict[tuple[str, str], int] = {
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/2_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/3_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/3_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/4_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/4_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/4_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/5_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/5_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/5_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/5_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/6_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/6_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/6_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/6_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/6_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/7_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/8_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/9_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/10_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/11_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/12_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/13_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/14_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/15_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/15_paper.xml", "data/paper_slides_data/raw_data/papers/16_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/15_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/16_paper.xml", "data/paper_slides_data/raw_data/papers/17_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/15_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/16_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/17_paper.xml", "data/paper_slides_data/raw_data/papers/18_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/15_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/16_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/17_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/18_paper.xml", "data/paper_slides_data/raw_data/papers/19_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/1_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/2_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/3_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/4_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/5_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/6_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/7_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/8_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/9_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/10_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/11_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/12_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/13_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/14_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/15_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/16_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0,
    ("data/paper_slides_data/raw_data/papers/17_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/18_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 1,
    ("data/paper_slides_data/raw_data/papers/19_paper.xml", "data/paper_slides_data/raw_data/papers/20_paper.xml"): 0
}