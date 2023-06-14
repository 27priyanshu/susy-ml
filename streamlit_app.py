import streamlit as st
import pandas as pd
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

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the CSV file using pandas
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
