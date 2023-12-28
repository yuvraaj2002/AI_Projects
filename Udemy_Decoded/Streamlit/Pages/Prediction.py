import streamlit as st

def prediction_module():
    st.markdown("<h1 style='text-align: center;font-size: 70px;'>Prediction Module</h1>", unsafe_allow_html=True)
    st.markdown("***")
    # st.write("")

    prediction_col1, prediction_col2 = st.columns([0.8, 2], gap="large")

    with prediction_col1:
        st.markdown("<h3>How to get prediction 🤷‍♂️</h3>", unsafe_allow_html=True)
        st.markdown("<p style='background-color: #C5FFF8; padding: 20px; border-radius: 10px; font-size: 20px;'>Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has ss with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum</p>", unsafe_allow_html=True)

        # Adjusting button width using CSS
        prediction_bt = st.button("Predict", use_container_width = True)  # Set the button width to 100% of the column
        if prediction_bt:
            model_output = st.toggle('Raw Model Output')
            if model_output:
                st.write("34")


    with prediction_col2:
        #st.markdown("<h3>How to get prediction 🤷‍♂️</h3>", unsafe_allow_html=True)
        number = st.number_input('How many number of lectures will be there in your course ?')
        content_duration = st.slider("What would be the duration of the content")
        option = st.selectbox(
            "Will your course be paid?",
            ("Yes course will be paid", "No it will be free"),
            index=None,
            placeholder="Select contact method...",
        )
        reviews = st.slider("Based on the quality of you course how many number of people you expect would give reviews? ")
        option = st.selectbox(
            "Does your course title contains any of these keywords (Beginner, Learn, Course)?",
            ("Yes it does", "No there are some other words"),
            index=None,
            placeholder="Select contact method...",
        )

        business = st.selectbox(
            "What is the main theme of your course",
            ("Business or Fianance", "Coding","Creativity"),
            index=None,
            placeholder="Select contact method...",
        )


        pass