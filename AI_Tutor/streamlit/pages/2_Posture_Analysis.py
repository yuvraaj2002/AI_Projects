import streamlit as st

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.0rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def posture_analysis_page():
    st.markdown(
        "<h1 style='text-align: left; font-size: 60px;'>Posture analysis🕵️</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 22px; text-align: left;padding-right: 2rem;padding-bottom: 1rem;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two powerful AI models thoroughly analyze the provided information. Once completed, users receive a clear indication of whether the job posting is is genuine or potentially decepti.</p>",
        unsafe_allow_html=True,
    )

    input_col,configuration_col  = st.columns(spec=(2,1.5), gap="large")
    with input_col:
        pass

    with configuration_col:
        video = st.file_uploader("Upload the video")
        st.write("***")
        row = st.columns(4)
        index = 0
        for col in row:
            tile = col.container(height=200)  # Adjust the height as needed
            tile.markdown(
                "<p style='text-align: left; font-size: 18px; '>This</p>",
                unsafe_allow_html=True,
            )
            index = index + 1

        video_download_col,statistics_download_col = st.columns(spec=(1,1), gap="large")
        with video_download_col:
            orignal_video = st.button("Play original video",use_container_width=True)
            if orignal_video:
                with input_col:
                    st.video(video)

            st.button("Download analysis chart", use_container_width=True)
        with statistics_download_col:
            processed_video = st.button("Play processed video",use_container_width=True)
            if processed_video:
                with input_col:
                    st.video(video)
            st.button("Download processed video", use_container_width=True)







posture_analysis_page()