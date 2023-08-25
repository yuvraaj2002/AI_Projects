import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns


st.title("Attributes of dataset")
st.write("""
    * **Sex:** The sex of the abalone, either male, female, or infant.
    * **Length:** The longest shell measurement in millimeters.
    * **Diameter:** The perpendicular shell measurement in millimeters.
    * **Height:** The height of the abalone with meat in the shell in millimeters.
    * **Whole weight:** The weight of the whole abalone in grams.
    * **Shucked weight:** The weight of the abalone meat in grams.
    * **Viscera weight:** The weight of the abalone viscera (internal organs) in grams.
    * **Shell weight:** The weight of the abalone shell in grams.
    * **Rings:** The number of rings on the abalone shell, which is used to determine the abalone's age.
    """)
df = pd.read_csv('Artifacts/Raw.csv')
st.dataframe(df.head(5))

st.subheader("Class distribution")
sex_counts = df['Sex'].value_counts()
labels = ['Male','Female','Infant']
sizes = sex_counts.values
total = sum(sizes)
percentages = [(size / total) * 100 for size in sizes]

custom_colors = ['#D7A1F9', '#B24BF3', '#880ED4']
pie_chart = px.pie(values=percentages, names=labels,color_discrete_sequence=custom_colors )

# Display the pie chart in Streamlit
st.plotly_chart(pie_chart)
