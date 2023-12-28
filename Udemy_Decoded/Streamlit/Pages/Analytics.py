import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def analytics_module():
    st.markdown("<h1 style='text-align: center;font-size: 70px;'>Analytics Module🔎</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "<p style='text-align: center; font-size: 25px; max-width: 1500px; margin: 0 auto;'>Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")

    Into_block = st.empty()
    columns = Into_block.columns([1, 1, 1])


    # Plotting a smaller pie chart in the first column using Plotly
    with columns[0]:

        # Hypothetical data for the pie charts
        labels = ['A', 'B', 'C', 'D']
        sizes_1 = [25, 35, 20, 20]
        sizes_2 = [20, 30, 25, 25]
        fig_1 = px.pie(values=sizes_1, names=labels)
        fig_1.update_traces(textposition='inside', textinfo='percent+label')
        fig_1.update_layout(height=400, width=400)  # Adjust size here
        st.plotly_chart(fig_1, use_container_width=True)

    # Plotting a smaller pie chart in the second column using Plotly
    with columns[1]:
        # Generating hypothetical data with a normal distribution
        np.random.seed(0)
        data = np.random.normal(loc=0, scale=1, size=1000)
        fig_hist = px.histogram(x=data, nbins=60)
        st.plotly_chart(fig_hist, use_container_width=True)

    with columns[2]:
        heatmap_data = np.random.rand(10, 10)
        fig_heatmap = px.imshow(heatmap_data, labels=dict(x="X-axis", y="Y-axis"), x=[f"X{i}" for i in range(10)],
                                y=[f"Y{i}" for i in range(10)], color_continuous_scale='Blues')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    st.write("<hr>", unsafe_allow_html=True)


    # Columns for the EDA
    EDA_col1,EDA_col2 = st.columns([1,1])

    with EDA_col1:
        st.markdown("<h3>What is percentage distribution of paid and unpaid courses?</h3>", unsafe_allow_html=True)
        st.title("")

        st.markdown("<h3>Among unpaid courses what are top 5 most popular courses?</h1>", unsafe_allow_html=True)
        st.title("")

        st.markdown("<h3>What is average price of course each category ?</h1>", unsafe_allow_html=True)

    with EDA_col2:
        st.markdown("<h3>Which type of course is most popular ?</h1>", unsafe_allow_html=True)

        st.markdown("<h3>What is average price of course each category ?</h1>", unsafe_allow_html=True)

        st.markdown("<h3>What is average content duration among paid and unpaid courses ?</h1>", unsafe_allow_html=True)
