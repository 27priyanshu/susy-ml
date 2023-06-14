import streamlit as st
import pandas as pd
import zipfile
import joblib
import os

# Define the path to the model file
model_file_path = os.path.join(os.getcwd(), "model.pkl")

# Load the logistic regression model
try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    st.error("Model file not found. Please make sure the model file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

# Define the Streamlit app
def main():
    st.title("Susy ML App")
    st.write("Welcome to the Susy ML App!")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "zip"])

    if uploaded_file is not None:
        # Check if the file is a zip file
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    # Extract the CSV file from the zip
                    csv_filename = zip_ref.namelist()[0]
                    zip_ref.extract(csv_filename)
                    # Read the CSV file using pandas
                    df = pd.read_csv(csv_filename)
            except Exception as e:
                st.error(f"Error processing the zip file: {str(e)}")
                st.stop()

        else:
            # Read the CSV file directly using pandas
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading the CSV file: {str(e)}")
                st.stop()

        # Perform prediction using the loaded model
        prediction = model.predict(df)

        # Display the prediction result
        st.subheader("Prediction Results")
        st.write(prediction)

if __name__ == "__main__":
    main()
