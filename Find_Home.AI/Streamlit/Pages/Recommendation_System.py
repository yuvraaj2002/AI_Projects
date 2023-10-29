import streamlit as st
import plotly.express as px
import pandas as pd
from Recommendation_Systems.Facility_RE import facilities_recommend_properties

location_list = ['Smartworld One DXP', 'M3M Crown', 'Adani Brahma Samsara Vilasa',
       'Sobha City', 'Signature Global City 93', 'Whiteland The Aspen',
       'Bestech Altura', 'Elan The Presidential',
       'Signature Global City 92', 'Emaar Digihomes',
       'Signature Global City 79B', 'DLF The Arbour', 'M3M Antalya Hills',
       'Signature Global City 81', 'SS Linden Floors',
       'Mahindra Luminare', 'M3M Golf Hills', 'Suncity Vatsal Valley',
       'Whiteland Blissville', 'Trump Tower', 'Tulip Monsella',
       'Krisumi Waterfall Residences', 'M3M Capital', 'Godrej Meridien',
       'La Vida by Tata Housing', 'Birla Navya', 'Signature Global City',
       'Godrej 101', 'M3M Soulitude', 'BPTP Terra', 'M3M Skycity',
       'MRG The Crown', 'Godrej Nature Plus Serenity', 'SS The Leaf',
       'Eldeco Acclaim', 'Emaar Gurgaon Greens', 'Oxirich Chintamanis',
       'DLF Garden City Floors', 'Anant Raj Estates', 'Tulip Yellow',
       'BPTP Amstoria', 'Emaar Emerald Hills', 'M3M Golfestate',
       'ATS Triumph', 'ATS Marigold', 'Signature Global City 37D Ph 2',
       'DLF Alameda', 'Experion Windchants', 'Saan Verdante',
       '4S Aradhya Homes', 'Yash Vihar', 'Smart World Orchard',
       'DLF The Camellias', 'Birla Navya Avik', 'Adani Samsara Avasa',
       'DLF The Crest', 'DLF The Magnolias', 'DLF The Aralias',
       'Ansal API Esencia', 'Pioneer Araya', 'M3M Merlin',
       'Smart World Gems', 'Vatika Aspiration', 'Ace Palm Floors',
       'DLF Gardencity Enclave', 'Emaar Palm Heights',
       'Signature Global Park', 'Emaar MGF Marbella',
       'Rishali Luxe Residency 112', 'Puri The Aravallis',
       'International City by SOBHA Phase 2', 'Emaar MGF The Palm Drive',
       'BPTP Green Oaks', 'Puri Emerald Bay', 'Ireo Victory Valley',
       'DLF Gardencity', 'Tata Primanti', 'DLF Park Place',
       'Central Park Flower Valley', 'Ireo Skyon',
       'AIPL The Peaceful Homes', 'Adani M2K Oyster Grande', 'G99',
       'Emaar MGF Emerald Floors Premier', 'ROF Insignia Park',
       'DLF The Ultima', 'Indiabulls Enigma', 'Experion The Westerlies',
       'Hero Homes', 'Central Park Flower Valley Mikasa Plots',
       'M3M Skywalk', 'Ireo The Grand Arch', 'JMS The Nation',
       'Imperia The Esfera', 'Ramprastha Primera',
       'Experion The Heartsong', 'DLF New Town Heights 2',
       'DLF The Primus', 'DLF The Skycourt', 'Central Park Resorts',
       'Suncity Avenue 76', 'International City by Sobha Phase 1',
       'Ambience Creacions', 'Vatika Xpressions', 'M3M Sierra 68',
       'Anand Niketan', 'DLF The Belaire', 'Godrej Aria',
       'Ansals Shiva Som Valley', 'Vipul World',
       'Central Park Flower Valley Aqua Front Towers', 'Tulip Violet',
       'Eldeco Accolade', 'M3M Natura', 'Emaar Imperial Gardens',
       'Ireo City Plots', 'Parsvnath Exotica', 'Pioneer Urban Presidia',
       'Suncity Platinum Towers', 'Godrej Nature Plus',
       'Bestech Park View Grand Spa', 'Shree Vardhman Victoria',
       'Silverglades The Melia', 'Shree Vardhman Flora',
       'Vatika Seven Elements', 'Bellavista Central Park Resorts',
       'M3M Heights', 'Godrej Habitat', 'Adani Brahma Samsara',
       'DLF The Grove', 'Corona Optus',
       'Central Park Flower Valley Flamingo Floors',
       'ROF Insignia Park 2', 'Indiabulls Centrum Park', 'BPTP Fortuna',
       'Bestech Park View Spa Next', 'DLF The Pinnacle', 'Godrej Oasis',
       'Anant Raj Estate Plots', 'Mapsko The Icon 79',
       'DLF Regal Gardens', 'DLF The Icon', 'Vatika Sovereign Park',
       'Vatika Sovereign Next', 'Central Park Flower Valley The Room',
       'M3M Sky Lofts', 'Golden Park', 'Ireo Savannah',
       'Satya Merano Greens', 'ATS Kocoon', 'Paras Quartier',
       'Ashiana Amarah', 'JMS Prime Land', 'India Rashtra',
       'Vipul Tatvam Villa', 'Orris Woodview Residencies',
       'Emaar MGF Palm Hills', 'Vatika City', 'DLF New Town Heights 1',
       'Vatika Gurgaon 21', 'Signature The Roselia',
       'Vatika Independent Floors', 'Adani Tatva Estates',
       'Emaar Palm Gardens', 'Pareena Mi Casa', 'The Close North',
       'Emaar The Palm Springs', 'BPTP Park Serene', 'Orchid IVY Floors',
       'ILD Greens', 'Godrej Icon', 'Orris Aster Court Premier',
       'M3M Latitude', 'Emaar MGF Emerald Estate', 'Green Court',
       'TARC Maceo', 'Raheja Vanya', 'Paras Ekam Homes',
       'Landmark The Homes 81', 'ROF Normanton Park', 'Corona Greens',
       'Umang Winter Hills', 'Puri Diplomatic Greens',
       'Silverglades Hightown Residences', 'Pioneer Park',
       'Anant Raj Ashok Estate', 'Paras Dews', 'Ireo The Corridors',
       'Assotech Blith', 'Bestech Park View Sanskruti',
       'Signature Global the Millennia', 'Orchid Island',
       'Ramprastha The Edge Towers', 'Pyramid Spring Valley',
       'Bestech Park View Ananda', 'Mapsko Casa Bella', 'Mahindra Aura',
       'Godrej Air', 'Conscient Habitat', 'Conscient Heritage Max',
       'Vipul Belmonte', 'Unitech The Residences', 'ILD Grand',
       'Signature Global Solera 2', 'Signature Global Solera',
       'M3M Woodshire', 'Vatika India Next Plots',
       'MV Buildcon Precore City', 'Lion Infra Green Valley',
       'Orchid Petals', 'BPTP Mansions Park Prime',
       'Emaar MGF Palm Terraces', 'Optimal ultra luxury builder floors',
       'Salcon The Verandas', 'BPTP Park Generations', 'Zara Aavaas',
       'Yashika 104', 'Breez Global Heights 89', 'Zara Rossa',
       'Alpha Corp GurgaonOne 84', 'Krrish Florence Estate',
       'Tulip Purple', 'Tulip Ivory', 'Shree Vardhman City',
       'Signature Global Prime', 'Antriksh Heights', 'BPTP Pedestal',
       'Vatika Express City', 'Pegasus Atulyam 83', 'DLF The Summit',
       'The Close South', 'Emaar Mgf Palm Terraces Select',
       'Unitech Fresco', 'Unitech Escape', 'Unitech Harmony',
       'Vatika The Seven Lamps', 'BPTP Freedom Park Life',
       'DLF New Town Heights', 'La Lagune', 'M3M My Den',
       'Suncity Avenue 102', 'DLF Princeton Estate',
       'Pyramid Urban Homes 2', 'Satya The Hermitage', 'BPTP Spacio',
       'SS The Coralwood']

def Recommendation_System_Page():

    page_col1, page_col2 = st.columns(spec=(1.8, 2.0), gap="large")
    with page_col1:
        st.title("Recommendation Configuration 🔩")
        html_text = ('<p style="font-size: 20px;">This recommendation system comprises a fusion of three distinct recommendation engines: Facilities-based recommendations, Price-based recommendations, and Nearby Locations recommendations. The ultimate recommendation is derived from the collective outcomes of these three recommendation systems. To assign greater significance to a specific recommendation system, you have the flexibility to adjust the weighting percentage below.</p>')
        st.markdown(html_text, unsafe_allow_html=True)
        st.write("")

        configuation_col, input_col = st.columns(spec=(1,1), gap="large")
        with configuation_col:

            # Input for the Facilities based recommendation system weight
            facilities_recommendation_wt = st.slider("Select the Weightage of Facilities based recommendation system (%)", min_value=1, max_value=100, value=30, step=1,key='facilities_recommendation_wt')
            price_recommendation_wt = st.slider("Select the Weightage of Price based recommendation system (%)", min_value=1, max_value=100, value=30, step=1,key='price_recommendation_wt')
            location_recommendation_wt = st.slider("Select the Weightage of Location based recommendation system (%)", min_value=1, max_value=100, value=30, step=1,key='location_recommendation_wt')

        with input_col:

            # Create data for the pie chart
            data = {
                'Categories': ['Facilities', 'Price', 'Location'],
                'Weights': [facilities_recommendation_wt, price_recommendation_wt, location_recommendation_wt]
            }

            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            custom_colors = ["#AEF359", "#03C04A", "#0B6623"]

            # Create a dynamic pie chart using Plotly Express
            fig = px.pie(df, names='Categories', values='Weights', title='Recommendation System Weights',color_discrete_sequence=custom_colors)
            st.plotly_chart(fig, use_container_width=True)


    with page_col2:
        st.title("Enter Location 🏡")
        user_input = st.selectbox("Select any location from pool of 256 unique locations in gurgaon", location_list, index=None,
                     placeholder="Select Location for which you want to get recommendations", key='user_input')
        st.markdown("***")

        # Three different sections for the top 5 recommendation from each of the recommendation system
        facilities_col, price_col, location_col = st.columns(spec=(1, 1, 1), gap="large")

        with facilities_col:
            facilities_recommendations = st.toggle('Show Facilities based recommendations')
            if facilities_recommendations:
                facilities_results = facilities_recommend_properties(user_input)
                facilities_results = facilities_results['PropertyName'].values
                st.write(facilities_results)

        with price_col:
            price_recommendations = st.toggle('Show Price based recommendations')
            if price_recommendations:
                st.write('Feature activated!')

        with location_col:
            locations_recommendations = st.toggle('Show Location based recommendations')
            if locations_recommendations:
                st.write('Feature activated!')

        st.title("Combined recommendations")