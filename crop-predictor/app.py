from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [float(request.form[x]) for x in ['nitrogen', 'phosphorus', 'potassium',
                                                     'temperature', 'humidity', 'ph', 'rainfall']]
        prediction = model.predict([np.array(features)])
        return render_template('index.html', prediction_text=f"Recommended Crop: {prediction[0]}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
