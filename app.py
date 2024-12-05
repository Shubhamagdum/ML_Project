from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('D:\ML_Project\Dataset\Algerian_forest_fires_cleaned_dataset.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    day = int(request.form['day'])
    month = int(request.form['month'])
    year = int(request.form['year'])
    # ... (other input fields)

    # Create a list of input features
    input_features = [day, month, year, temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region]

    # Preprocess the input features (e.g., one-hot encoding for categorical features)
    # ... (Implement preprocessing steps)

    # Scale the input features using the same scaler used during training
    input_features_scaled = scaler.transform([input_features])

    # Make prediction using the model
    prediction = model.predict(input_features_scaled)

    return render_template('home.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)