import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import numpy as np

# Define your preprocessor (moved to global scope)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Passenger Class', 'Age', 'SibSp', 'Parch', 'Fare']),
        # ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Embarked'])
    ])


def save_preprocessor():
    """
    Save the preprocessor to a file.

    This function creates a directory for the model if it doesn't exist,
    then saves the preprocessor object to a pickle file named 'preprocessor.pkl'
    in the model directory.

    Parameters:
    None

    Returns:
    None

    Side Effects:
    - Creates a directory for the model if it doesn't exist.
    - Saves the preprocessor object to a pickle file named 'preprocessor.pkl'
      in the model directory.
    - Prints a message indicating the successful saving of the preprocessor.
    """
    model_directory = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_directory, exist_ok=True)

    preprocessor_path = os.path.join(model_directory, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")


def load_model(path):
    """Load a trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def data_preprocessing(data):
    """Preprocess input data."""
    X = pd.DataFrame(data['titanic'], index=[0])
    X = X[['Passenger Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Passenger Class', 'Age', 'SibSp', 'Parch', 'Fare']),
        # ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Embarked'])  
    ])
    
    processed = preprocessor.fit_transform(X)
    
    if X["Sex"].iloc[0] == "male":
        processed = np.append(processed, [1, 0])
    elif X["Sex"].iloc[0] == "female":
        processed = np.append(processed, [0, 1])
    else:
        pass
    
    if X["Embarked"].iloc[0] == "S":
        processed = np.append(processed, [1, 0, 0])
    elif X["Embarked"].iloc[0] == "C":
        processed = np.append(processed, [0, 1, 0])
    elif X["Embarked"].iloc[0] == "Q":
        processed = np.append(processed, [0, 0, 1])
    else:
        pass
    
    # Transform and return as numpy array
    print(f"processed features: {processed}")
    return processed
    