from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_filename = 'bagging_classifier_model.pickle'
with open(model_filename, 'rb') as model_file:
    modelku = pickle.load(model_file)

# Load the pre-processing scaler
scaler_filename = 'preprocessing.pickle'
with open(scaler_filename, 'rb') as scaler_file:
    pra_proses = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request.
        data = request.get_json(force=True)
        
        # Convert categorical features to numeric values
        categorical_features = {
            "sex": {"male": 1, "female": 0},
            "cp": {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3},
            "fbs": {"true": 1, "false": 0},
            "restecg": {"normal": 0, "having ST-T wave abnormality": 1, "showing probable or definite left ventricular hypertrophy": 2},
            "exang": {"yes": 1, "no": 0},
            "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
            "thal": {"normal": 1, "fixed defect": 2, "reversable defect": 3}
        }

        for feature, mapping in categorical_features.items():
            data[feature] = mapping[data[feature]]

        # Convert the data into a DataFrame
        data_df = pd.DataFrame([data])

        # Pre-process the data
        transformed_data = pra_proses.transform(data_df)

        # Make prediction
        prediction = modelku.predict(transformed_data)

        # Return the prediction as a JSON response
        prediction_text = "No Heart Disease" if int(prediction[0]) == 0 else "Heart Disease"
        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
