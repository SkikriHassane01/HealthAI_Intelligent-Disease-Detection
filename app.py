# TODO:______________________________________________________importing the libraries______________________________________________________

from flask import Flask, request, redirect, url_for, render_template
from numpy import array,nan,expand_dims,argmax
from pandas import DataFrame
import os 
import joblib, pickle
from werkzeug.utils import secure_filename 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from FeatureEngineeringDiabetes import BMITransformer, InsulinTransformer, GlucoseTransformer
# TODO:______________________________________________________creating and configuring the flask app______________________________________________________

# Global variables:
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
models = {
    'alzheimer': load_model('Models/alzheimer_model.keras'),
    'brain_tumor': load_model('Models/Brain_tumor_model.keras'),
    'breast_cancer': pickle.load(open('Models/breast_cancer_model.pkl', 'rb')),
    'covid19': load_model('Models/covid_model.h5', compile=False),
    'diabetes': joblib.load('Models/diabetes_model.pkl'),
    'pneumonia': load_model('Models/pneumia_model.keras')
}

model = models['covid19']
# Re-compile the model with the correct settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TODO:______________________________________________________Utility Functions______________________________________________________
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(file_path, disease, height=150, width=150, color_mode='grayscale'):
    img = load_img(file_path, target_size=(height, width), color_mode=color_mode)
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, axis=0)
    
    model = models[disease]
    prediction = model.predict(img_array)
    result = argmax(prediction, axis=1)[0]
    return result

# TODO:______________________________________________________Define the Home and prediction Routes______________________________________________________

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<disease>', methods=['GET', 'POST'])
def disease_page(disease):
    prediction_result = None
    if disease in ['alzheimer', 'brain_tumor', 'covid19', 'pneumonia', 'BreastCan', 'diabetes']:
        if request.method == 'POST':
            if disease in ['alzheimer', 'brain_tumor', 'covid19', 'pneumonia']:
                prediction_result = handle_image_disease(request, disease)
            elif disease == 'BreastCan':
                prediction_result = handle_breast_cancer(request)
            elif disease == 'diabetes':
                prediction_result = handle_diabetes(request)
        return render_template(f'{disease}.html', prediction_result=prediction_result)
    return redirect(url_for('index'))

# TODO:______________________________________________________Handle image-based Disease prediction______________________________________________________

def handle_image_disease(request, disease):
    if 'image' not in request.files:
        return {'error': 'No file part'}
    file = request.files['image']
    if file.filename == '':
        return {'error': 'No selected file'}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if disease == 'alzheimer':
            result = transform_image(file_path, 'alzheimer', 176, 208, 'rgb')
            alzheimer_classes = {
                0: 'MildDemented',
                1: 'ModerateDemented',
                2: 'NonDemented',
                3: 'VeryMildDemented'
            }
            result = alzheimer_classes.get(result, 'Unknown')
        elif disease == 'brain_tumor':
            result = transform_image(file_path, 'brain_tumor', 150, 150, 'rgb')
            brain_classes = {
                0: 'glioma',
                1: 'meningioma',
                2: 'no tumor',
                3: 'pituitary'
            }
            result = brain_classes.get(result, 'Unknown')
        elif disease == 'covid19':
            result = transform_image(file_path, 'covid19', 70, 70, 'rgb')
            covid_classes = {
                0: 'Normal',
                1: 'COVID',
                2: 'Lung_Opacity',
                3: 'Viral Pneumonia'
            }
            result = covid_classes.get(result, 'Unknown')
        elif disease == 'pneumonia':
            result = transform_image(file_path, 'pneumonia', 150, 150)
            pneumonia_classes = {
                0: 'Pneumonia',
                1: 'Normal'
            }
            result = pneumonia_classes.get(result, 'Unknown')

        categorized_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], disease, result)
        if not os.path.exists(categorized_folder_path):
            os.makedirs(categorized_folder_path)

        unique_filename = filename
        counter = 1
        while os.path.exists(os.path.join(categorized_folder_path, unique_filename)):
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1

        categorized_file_path = os.path.join(categorized_folder_path, unique_filename)
        os.rename(file_path, categorized_file_path)

        return {'prediction': result, 'file_path': categorized_file_path.replace('\\', '/')}
    return {'error': 'File not allowed'}

# TODO:______________________________________________________Handle Breast Cancer Prediction______________________________________________________
def handle_breast_cancer(request):
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
                'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    
    data = {feature: float(request.form.get(feature)) for feature in features}
    df = DataFrame(data, index=[0])
    
    model = models['breast_cancer']
    result = model.predict(df)[0]
    cancer_classes = {0: 'Benign', 1: 'Malignant'}
    return {'prediction': cancer_classes.get(result, 'Unknown')}

# TODO:______________________________________________________Handle Diabetes Prediction______________________________________________________
def handle_diabetes(request):
    input_features = {
        'Pregnancies': int(request.form.get('Pregnancies')),
        'Glucose': float(request.form.get('Glucose')),
        'BloodPressure': float(request.form.get('BloodPressure')),
        'SkinThickness': float(request.form.get('SkinThickness')),
        'Insulin': float(request.form.get('Insulin')),
        'BMI': float(request.form.get('BMI')),
        'DiabetesPedigreeFunction': float(request.form.get('DiabetesPedigreeFunction')),
        'Age': int(request.form.get('Age')),
    }

    df = DataFrame(input_features, index=[0])
    df = BMITransformer().transform(df)
    df = InsulinTransformer().transform(df)
    df = GlucoseTransformer().transform(df)
    
    model = models['diabetes']
    result = model.predict(df)[0]
    diabetes_classes = {0: 'No Diabetes', 1: 'Diabetes'}
    return {'prediction': diabetes_classes.get(result, 'Unknown')}

# TODO:______________________________________________________Running the flask application______________________________________________________
if __name__ == '__main__':
    app.run(debug=True)
