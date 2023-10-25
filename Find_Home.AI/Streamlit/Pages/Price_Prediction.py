import streamlit as st

property_type_options = ("Email", "Home phone", "Mobile phone")
agePossession_options = ("Email", "Home phone", "Mobile phone")
luxury_category_options = ("Email", "Home phone", "Mobile phone")
floor_category_options = ("Email", "Home phone", "Mobile phone")

def Price_Prediction_Page():

    page_col1, page_col2 = st.columns(spec=(2, 1.6), gap="large")
    with page_col1:
        st.markdown("<h1 class='center' style='font-size: 50px;'>Home Finder.AI</h1>", unsafe_allow_html=True)

        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")

        with input_col1:
            Property_Type = st.selectbox("Select Property Type",property_type_options,index=None,placeholder="Select",key='property_type_input')
            bathroom = st.slider("Select Number of Bathroom:", min_value=0, max_value=10, value=5, step=1,key='bathroom_input')
            agePossession = st.selectbox("Select Property Type",agePossession_options, index=None,placeholder="Select.",key='agePoss_input')
            furnishing_type = st.selectbox("Select Floor Category", floor_category_options, index=None,placeholder="Select.", key='furnishing_input')
            built_up_area = st.slider("Select Number of Bathroom:", min_value=0, max_value=10, value=5, step=1,key='builtArea_input')
            sector = st.selectbox("Select Floor Category", floor_category_options, index=None,placeholder="Select.", key='sector_input')

        with input_col2:
            bedRoom = st.slider("Select Number of Bedrooms:", min_value=0, max_value=10, value=5, step=1,key='bedroom_input')
            luxury_category = st.selectbox("Select Luxury Category", luxury_category_options, index=None,placeholder="Select.", key='luxury_category_input')
            floor_category = st.selectbox("Select Floor Category", floor_category_options,index=None, placeholder="Select.", key='floor_category_input')
            servant_room = st.selectbox("Select Floor Category", floor_category_options, index=None,placeholder="Select.", key='servent_input')
            balcony = st.selectbox("Select Floor Category", floor_category_options, index=None,placeholder="Select.", key='balcony_input')



    with page_col2:
        st.markdown("<h1 class='center' style='font-size: 50px;'>Home Finder.AI</h1>", unsafe_allow_html=True)
