import streamlit as st
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import io
import nltk
nltk.download('stopwords')
import string
nltk.download('punkt')
punctuation = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))


@st.cache_resource
def load_model():
    llm = CTransformers(model = '/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/Artifacts/llama-2-7b.ggmlv3.q4_1.bin',
                        model_type = 'llama',
                        config = {'temperature':0.5})
    return llm


def llama_response(Input_text):

    # Getting the model
    llm_model = load_model()

    # Defining the prompt template
    Que_template = PromptTemplate(

        input_variables=['data'],
        template="""
        Given a candidate's profile stored in the provided {data} text, automatically extract information about their projects, experience, and achievements.
        Utilize this extracted information to craft 10 interview questions that an interviewer might ask during the hiring process.

        **Data Processing:**

        - Employ text processing techniques to identify and segment relevant sections in the data, such as "Experience", "Projects", and "Achievements".
        - Extract key entities and information from each section, including:
            - **Experience:** Company, job title, duration, responsibilities, key skills used.
            - **Projects:** Project title, description, skills used, achievements.
            - **Achievements:** Awards, recognitions, notable accomplishments.

        **Question Generation:**

        - Analyze the extracted information to identify relevant details, skills, and achievements that could be explored through questions.
        - Consider the desired focus area (technical, behavioral, or a mix) and adjust the questions accordingly.
        - Include a variety of question types (open-ended, closed-ended, situational, hypothetical) to create a well-rounded set.
        - Arrange the questions in a logical order, starting with easier prompts and gradually increasing in complexity.

        **Example Prompts:**

        - (Based on experience) Tell me about a challenging situation you faced at [company name] and how you resolved it. What key skills did you leverage?
        - (Based on projects) Describe your role in [project title] and a specific technical challenge you encountered. How did you overcome it?
        - (Based on achievements) Your achievement of [achievement] is impressive. Can you elaborate on the context and the skills you used to achieve it?
        - (Based on skills) You mentioned proficiency in [skill]. Can you explain a situation where you applied this skill to achieve a positive outcome?

        **Remember:**
        - Tailor the questions to the specific information extracted from the data and the desired job role.
        - Emphasize natural language generation techniques to create grammatically correct and well-structured questions.

        """
    )

    model_response = llm_model(Que_template.format(data = Input_text))
    return model_response


def clean_text(text):
    """
    This function preprocesses text by performing the following steps:

    1. **Lowercasing:** Converts all characters to lowercase.
    2. **Tokenization:** Splits the text into individual words using NLTK word tokenization.
    3. **Stopword removal:** Removes common stop words (e.g., "the", "a", "is") from the word list.
    4. **Optional replacement:** Optionally replaces stop words with empty strings (default behavior).
    5. **Rejoining:** Joins the remaining words back into a string with spaces in between.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text with stop words removed or replaced.
    """

    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = ['' if word in stop_words else word for word in tokens]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def resume_radar_page():

    col1,col2 = st.columns(spec=(2, 1.5), gap="large")
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 50px; '>Resume Radar 👨‍💼</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>To enhance the precision of tailored recommendations from our recommendation system, accurately input details such as the expected price range you're considering and the names of the apartments you're interested in. This focused input enables our fusion of two distinct recommendation engines—Facilities-bnces and financial considerations, optimizing your search for the perfect apartment.</p>",
            unsafe_allow_html=True,
        )

    with col2:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:

            bytes_data = uploaded_file.read()
            file_object = io.BytesIO(bytes_data)  # Create a BytesIO object
            reader = PdfReader(file_object)
            # Extract text from all pages
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            processsed_text = clean_text(text)
            st.write(llama_response(processsed_text))

            # st.title("Resume")
            # with open(pdf_file, 'rb') as f:
            #     pdf_data = f.read()
            # b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            #
            # pdf_display = F'<embed src="data:application/pdf;base64,{b64_pdf}" width="700" height="700" type="application/pdf">'
            #
            # st.markdown(pdf_display, unsafe_allow_html=True)
