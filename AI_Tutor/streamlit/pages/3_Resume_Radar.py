import streamlit as st
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from nltk.stem import WordNetLemmatizer
import io
import string
import nltk
import numpy as np
from nltk.corpus import wordnet
import re

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


nltk.download("stopwords")
import string

nltk.download("punkt")
punctuation = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words("english"))


@st.cache_resource
def load_model():
    llm = CTransformers(
        model="/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/artifacts/llama-2-7b.ggmlv3.q4_1.bin",
        model_type="llama",
        config={"temperature": 0.5},
    )
    return llm


def llama_response(Input_text):

    # Getting the model
    llm_model = load_model()

    # Defining the prompt template
    Que_template = PromptTemplate(
        input_variables=["data"],
        template="""
        Given a candidate's profile stored in the provided {data} text, automatically extract information about their projects, experience, and achievements.
        Utilize this extracted information to craft 10 interview questions that an interviewer might ask during the hiring process.""",
    )

    model_response = llm_model(Que_template.format(data=Input_text))
    return model_response


def clean_data(text):
    """
    This function preprocesses text by performing the following steps:

    1. **Lowercasing:** Converts all characters to lowercase.
    2. **Tokenization:** Splits the text into individual words using NLTK word tokenization.
    3. **Stopword removal:** Removes common stop words (e.g., "the", "a", "is") from the word list.
    4. **Punctuation removal:** Removes punctuation marks from the word list.
    5. **Remove phone numbers and email addresses:** Removes phone numbers and email addresses from the text.
    6. **Lemmatization:** Lemmatizes words to their base form.
    7. **Rejoining:** Joins the remaining words back into a string with spaces in between.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text with stop words, punctuation, phone numbers, email addresses, and lemmatization applied.
    """
    # Lowercase the text
    text_lower = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text_lower)

    # Remove punctuation marks
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove phone numbers
    cleaned_tokens = []
    for token in tokens:
        if not re.match(r"\+?\d[\d -]{8,12}\d", token):
            cleaned_tokens.append(token)

    # Remove "gmail.com" after punctuation removal
    final_tokens = []
    for token in cleaned_tokens:
        if not token.endswith("gmail.com"):
            final_tokens.append(token)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in final_tokens]

    # Rejoin the tokens into a string
    cleaned_text = " ".join(lemma_tokens)
    return cleaned_text


# Define a function to get the vector representation of a document using Word2Vec
# def document_vector(doc):
#     # Remove out-of-vocabulary words and get Word2Vec vectors for the words in the document
#     words = [word for word in doc.split() if word in W2V_model]
#     if not words:
#         # If none of the words are in the Word2Vec model, return zeros
#         return np.zeros(300)
#
#     # Return the mean of Word2Vec vectors for words in the document
#     return np.mean(W2V_model[words], axis=0)
#
#
# # Apply the function to each document in Facilities_df['Facilities_Str']
# word2vec_matrix = np.array(
#     [document_vector(doc) for doc in Facilities_df["Facilities_Str"]]
# )


def process_clean(pdf_data):

    file_object = io.BytesIO(pdf_data)  # Create a BytesIO object
    reader = PdfReader(file_object)

    # Extract text from all pages
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return clean_data(text)


def resume_radar_page():

    col1, col2 = st.columns(spec=(2, 1.3), gap="large")
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 50px; '>Resume Radar 👨‍💼</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>To enhance the precision of tailored recommendations from our recommendation system, accurately input details such as the expected price range you're considering and the names of the apartments you're interested in. This focused input enables our fusion of two distinct recommendation engines—Facilities-bnces and financial considerations, optimizing your search for the perfect apartment.</p>",
            unsafe_allow_html=True,
        )
        job_description = st.text_input("Enter the job description")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    with col2:

        if uploaded_file is not None:
            pdf_data = uploaded_file.read()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="690" height="740" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.write("")
            analyze_bt = st.button("Analyze my resume🔎", use_container_width=True)
            if analyze_bt:
                with col1:
                    st.write(clean_data(job_description))
                    # st.write(process_clean(pdf_data))


resume_radar_page()
