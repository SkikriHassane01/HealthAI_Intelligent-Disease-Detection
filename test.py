# test_load_models.py
from keras.models import load_model
import joblib

# Test Keras model
try:
    keras_model = load_model('./Models/Brain_tumor_model.keras')
    print("Keras model loaded successfully!")
except Exception as e:
    print("Error loading Keras model:", e)

# Test Joblib model
try:
    diabetes_model = joblib.load('./Models/diabetes_model.pkl')
    print("Joblib model loaded successfully!")
except Exception as e:
    print("Error loading Joblib model:", e)
