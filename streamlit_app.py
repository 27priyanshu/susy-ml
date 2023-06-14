import streamlit as st
import pandas as pd
import zipfile
import joblib

# Load the logistic regression model
model = joblib.load('logistic_regression_model.pkl')
def main():
    st.title('Anomaly Detection')
    
    file = st.file_uploader('Upload CSV file or ZIP file', type=['csv', 'zip'])
    
    if file is not None:
        df = process_file(file)
        if df is not None:
            prediction = model.predict(df)
            st.write('Prediction:', prediction)

def process_file(file):
    if file.type == 'application/zip':
        try:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                csv_filename = zip_ref.namelist()[0]
                with zip_ref.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
        except:
            st.error('Invalid zip file')
            return None
    else:
        try:
            df = pd.read_csv(file)
        except:
            st.error('Invalid file format')
            return None
    
    return df

if __name__ == '__main__':
    main()
