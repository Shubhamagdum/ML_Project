from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('D:\\ML_Project\\Notebook\\fwi_model.pkl','rb'))
scaler = pickle.load(open('D:\\ML_Project\\Notebook\\scaler.pkl' ,'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GAT','POST'])
def predict_datapoint():
     result=""
     if request.method == 'POST':
        features = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['ISI']),
            int(request.form['Region'])
        ]
        scaled_features = scaler.transform([features])

    # Predict using model
        prediction = model.predict(scaled_features)[0]
        result = label_encoder.inverse_transform([prediction])[0]  # Decode the prediction

        return render_template('home.html', result=result)
    
    
if __name__ == '__main__':
    app.run(debug=True)
