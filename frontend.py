import streamlit as st
from keras.models import load_model
import utils as utils
import numpy as np

LSTM_RNN_model = load_model("best_model.h5")

st.title('Model Inference App')
st.write('Upload XML files to get predictions.')

# File upload widgets for XML files
paper_file_content = st.file_uploader("Upload Paper XML file", type="xml")
presentation_file_content = st.file_uploader("Upload Presentation XML file", type="xml")

if paper_file_content and presentation_file_content:
    # paper_file_content = paper_file_content.read()
    # presentation_file_content = presentation_file_content.read()

    model_name = st.selectbox("Select Model", ["Model A", "Model B", "Model C"])

    if st.button('Predict'):
        if model_name == "Model A":
            category, probabilities, paper_score = utils.predict_LSTM_RNN(paper_file_content, presentation_file_content)
        elif model_name == "Model B":
            category, probabilities, paper_score = utils.predict_logistic_regression(paper_file_content, presentation_file_content)
        else:
            category, probabilities, paper_score = utils.predict_transformer(paper_file_content, presentation_file_content)

        if paper_score is not None:
            st.title('This presentation was ' + category + ' representation of this paper')
            st.write('with a weighted similarity score of ' + str(paper_score))

            # Transpose the list of probabilities
            entailment_probs = [prob[0] for prob in probabilities]
            neutral_probs = [prob[1] for prob in probabilities]
            contradiction_probs = [-1 * prob[2] for prob in probabilities]
            combined_probs = [entailment_probs[i] + 0.5 * neutral_probs[i] for i in range(len(entailment_probs))]
            scaled_contradiction_probs = [
                contradiction_probs[i] + 0.5 * entailment_probs[i] if contradiction_probs[i] + 0.5 * entailment_probs[
                    i] < 0 else -0.1 for i in range(len(entailment_probs))]

            window_size = 3
            smoothed_entail_probs = np.convolve(entailment_probs, np.ones(window_size) / window_size, mode='valid')
            smoothed_neutral_probs = np.convolve(neutral_probs, np.ones(window_size) / window_size, mode='valid')
            smoothed_contradiction_probs = np.convolve(contradiction_probs, np.ones(window_size) / window_size,
                                                       mode='valid')
            smoothed_combined_probs = np.convolve(combined_probs, np.ones(window_size) / window_size,
                                                  mode='valid')
            smoothed_scaled_contradiction_probs = np.convolve(scaled_contradiction_probs,
                                                              np.ones(window_size) / window_size,
                                                              mode='valid')

            st.write('Sentence Entailment Probabilities for ', len(probabilities), 'sentences')

            # Display an area chart for entailment and contradiction probabilities
            # st.area_chart({"Entailment": smoothed_entail_probs, "Contradiction": smoothed_contradiction_probs},
            #               color=["#FF4118", "#18D6FF"],
            #               use_container_width=True)
            st.area_chart({"Entailment": smoothed_entail_probs, "Contradiction": smoothed_scaled_contradiction_probs},
                          color=["#FF4118", "#18D6FF"],
                          use_container_width=True)
        else:
            st.write('invalid slide/presentation XML, try again')
