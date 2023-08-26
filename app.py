# Importing Dependencies
import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Creating a Streamlit web-page
with st.sidebar:
    st.image(
        "https://imageio.forbes.com/specials-images/dam/imageserve/966248982/960x0.jpg?height=456&width=711&fit=bounds")
    st.title("Fully Automated ML model")
    choice = st.radio("Select your task", ["Upload", "Analyse", "Train", "Test"])

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

if choice == "Upload":
    st.title('Upload your file for modeling')
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Analyse":
    st.header("Exploratory Data Analysis")
    Analysed_DF = df.profile_report()
    st_profile_report(Analysed_DF)

if choice == "Train":
    st.title("Training your model")
    chosen_target = st.selectbox('Choose the target variable', df.columns)
    x = df.drop(chosen_target, axis=1)
    x.to_csv('newdf.csv', index=None)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button('Run Regression'):
            from pycaret.regression import setup, compare_models, pull, save_model

            setup(df, target=chosen_target)
            setup_df = pull()
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    with col2:
        if st.button('Run Classification'):
            from pycaret.classification import setup, compare_models, pull, save_model

            setup(df, target=chosen_target)
            setup_df = pull()
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

if os.path.exists('./newdf.csv'):
    new_df = pd.read_csv('newdf.csv', index_col=None)



if choice == "Test":
    st.title("Test your model")
    from pycaret.regression import load_model
    from pycaret.classification import load_model

    info = pd.DataFrame()
    for i in new_df.columns:
        test_i = st.selectbox(i, (new_df[i].unique()))
        info[i] = [test_i]

    if st.button('Submit'):
        st.write("This the data you have entered")
        st.dataframe(info)

        model = load_model('best_model')
        results = model.predict(info)
        st.write('This is your output')
        results
    
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
