from flask import Flask, request, jsonify, abort
import logging
import numpy as np
from src.utils import data_preprocessing, load_model
import os

app = Flask(__name__)

# Configuration
API_KEY = 'dfhsAddsf124*&&dcvv'
MODEL_PATH = os.path.join('model', 'model.pkl')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_inputs(sex, age, passenger_class, sibsp, parch, fare, embarked):
    """Validate all input parameters"""
    errors = []
    
    if sex.lower() not in ['male', 'female']:
        errors.append("Sex must be 'male' or 'female'")
    if not (0 < age <= 120):
        errors.append("Age must be between 1 and 120")
    if passenger_class not in [1, 2, 3]:
        errors.append("Passenger class must be 1, 2, or 3")
    if sibsp < 0:
        errors.append("SibSp cannot be negative")
    if parch < 0:
        errors.append("Parch cannot be negative")
    if fare <= 0:
        errors.append("Fare must be positive")
    if embarked.upper() not in ['S', 'C', 'Q']:
        errors.append("Embarked must be S, C, or Q")
    
    if errors:
        raise ValueError("; ".join(errors))


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error)}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": str(error)}), 401

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": str(error)}), 500

@app.route('/api/v1/get/predict', methods=['GET'])
def predict():
    """
    Main prediction endpoint for the Titanic survival model.

    This function handles GET requests to predict the survival of a Titanic passenger
    based on input parameters. It validates the API key, checks model readiness, 
    validates input parameters, processes the data, and returns the prediction.

    Parameters:
    None (parameters are extracted from the request arguments)

    Returns:
    Response: A JSON response containing the prediction result and its description.
              - "prediction": 1 if the passenger survived, 0 otherwise.
              - "description": Explanation of the prediction values.
    """
    logger.info('Prediction request received')

    # API Key validation
    api_key = request.headers.get('x-api-key')
    if not api_key or api_key != API_KEY:
        logger.warning("Unauthorized access attempt")
        abort(401, "Unauthorized - Invalid API Key")

    model = load_model(MODEL_PATH)
    # Extract and validate parameters
    # Check if model is ready
    if not model:
        logger.error("Model not initialized")
        abort(500, "Service unavailable - Model not loaded")

    try:
        # Extract and validate parameters
        params = {
            'sex': request.args.get('sex'),
            'age': request.args.get('age', type=float),
            'passenger_class': request.args.get('passenger_class', type=int),
            'sibsp': request.args.get('sibsp', type=int),
            'parch': request.args.get('parch', type=int),
            'fare': request.args.get('fare', type=float),
            'embarked': request.args.get('embarked')
        }

        if any(v is None for v in params.values()):
            missing = [k for k, v in params.items() if v is None]
            raise ValueError(f"Missing parameters: {', '.join(missing)}")

        validate_inputs(**params)

        # Prepare input data
        input_data = {
            "Passenger Class": params['passenger_class'],
            "Sex": params['sex'],
            "Age": params['age'],
            "SibSp": params['sibsp'],
            "Parch": params['parch'],
            "Fare": params['fare'],
            "Embarked": params['embarked']
        }

        # Process and predict
        processed_data = data_preprocessing({"titanic": input_data})
        prediction = model.predict(np.array([processed_data]))[0]

        return jsonify({
            "prediction": int(prediction),
            "description": "1 = Survived, 0 = Did not survive"
        })

    except ValueError as e:
        logger.warning(f"Bad request: {str(e)}")
        abort(400, str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        abort(500, "Internal server error")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the readiness of the model and preprocessor.

    This function checks if the model and preprocessor are loaded and returns their status.

    Returns:
    Response: A JSON response containing the status of the model and preprocessor.
              - "model_loaded": Boolean indicating if the model is loaded.
              - "preprocessor_loaded": Boolean indicating if the preprocessor is loaded.
              - "status": A string indicating readiness, "ready" if both are loaded, "not ready" otherwise.
    """
    status = {
        "model_loaded": bool(model),
        "status": "ready" if model else "not ready"
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(port=5001)