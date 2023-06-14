from flask import Flask, render_template, request
import pandas as pd
import zipfile
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the logistic regression model
model = LogisticRegression()
model.load_model('logistic_regression_model.pkl')  # Assuming you have saved the trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    # Check if the file is a zip file
    if file.filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                # Extract the CSV file from the zip
                csv_filename = zip_ref.namelist()[0]
                zip_ref.extract(csv_filename)
                # Read the CSV file using pandas
                df = pd.read_csv(csv_filename)
        except:
            return render_template('index.html', error='Invalid zip file')

    else:
        # Read the CSV file directly using pandas
        try:
            df = pd.read_csv(file)
        except:
            return render_template('index.html', error='Invalid file format')

    # Perform prediction using the loaded model
    prediction = model.predict(df)

    # Return the prediction result to the user
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
