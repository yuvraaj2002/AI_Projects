import streamlit as st
import pandas as pd
import time



def Add_questions_page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px; '>Recommendation Engine 👨‍💼</h1>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            "<p style='font-size: 20px; text-align: center;padding-left: 2rem;padding-right: 2rem;padding-bottom: 1rem;'>To enhance the precision of tailored recommendations from our recommendation system, accurately input details such as the expected price range you're considering and the names of the apartments you're interested in. This focused input enables our fusion of two distinct recommendation engines—Facilities-based and Price-based recommendations—to refine suggestions based on your budget and preferences. The ultimate recommendation derives from the collective outcomes of both systems, and you can further customize the significance of each by adjusting the weighting percentage below. This streamlined approach ensures that our recommendations align with your preferences and financial considerations, optimizing your search for the perfect apartment.</p>",
            unsafe_allow_html=True,
        )
        que_file = st.file_uploader("Upload the file containing questions")


    recommendation_col, weight_plot_col = st.columns(spec=(2.2, 1), gap="large")
    with recommendation_col:

            st.markdown(
                "<p style='font-size: 18px; padding-bottom: 1rem;'>Presenting the top five apartments meticulously curated for your consideration, derived from your selected apartment and the thoughtful configurations of our recommendation engine weights. We trust these recommendations will add value to your search and enhance your experience in finding the ideal residence.</p>",
                unsafe_allow_html=True,
            )
            # row = st.columns(5)
            # index = 0
            # for col in row:
            #     tile = col.container(height=200)  # Adjust the height as needed
            #     tile.markdown(
            #         "<p style='text-align: left; font-size: 18px; '>"
            #         + str(facilities_results["PropertyName"][index])
            #         + "</p>",
            #         unsafe_allow_html=True,
            #     )
            #     if index == 4:
            #         tile.metric(
            #             label="Similarity Score",
            #             value=round(facilities_results["SimilarityScore"][index], 3),
            #             delta="Base line score",
            #         )
            #     else:
            #         tile.metric(
            #             label="Similarity Score",
            #             value=round(facilities_results["SimilarityScore"][index], 3),
            #             delta=round(
            #                 facilities_results["SimilarityScore"][index]
            #                 - baseline_similarity_score,
            #                 5,
            #             ),
            #         )
            #     index = index + 1

            # st.write(facilities_results)

