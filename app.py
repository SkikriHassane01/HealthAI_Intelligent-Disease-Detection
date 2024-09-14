
# TODO:______________________________________________________importing the libraries______________________________________________________

from flask import Flask, request, redirect, url_for, render_template
from numpy import array,nan,expand_dims,argmax
from pandas import DataFrame,Series,to_numeric
import os 
import joblib, pickle
from werkzeug.utils import secure_filename 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.base import BaseEstimator, TransformerMixin

# TODO:______________________________________________________creating and configuring the flask app______________________________________________________


class BMITransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        NewBMI = Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
        X['NewBMI'] = "Normal"
        X['BMI'] = to_numeric(X['BMI'], errors='coerce')  # Convert BMI to numeric, setting errors to NaN
        X.loc[X["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
        X.loc[(X["BMI"] >= 18.5) & (X["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
        X.loc[(X["BMI"] > 24.9) & (X["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
        X.loc[(X["BMI"] > 29.9) & (X["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
        X.loc[(X["BMI"] > 34.9) & (X["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
        X.loc[X["BMI"] > 39.9, "NewBMI"] = NewBMI[5]
        return X

class InsulinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def set_insulin(row):
            if 16 <= row["Insulin"] <= 166:
                return "Normal"
            else:
                return "Abnormal"
        X["NewInsulinScore"] = X.apply(set_insulin, axis=1)
        return X

class GlucoseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.NewGlucose = Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype="category")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["NewGlucose"] = "Normal"
        X.loc[X["Glucose"] <= 70, "NewGlucose"] = self.NewGlucose[0]
        X.loc[(X["Glucose"] > 70) & (X["Glucose"] <= 99), "NewGlucose"] = self.NewGlucose[1]
        X.loc[(X["Glucose"] > 99) & (X["Glucose"] <= 126), "NewGlucose"] = self.NewGlucose[2]
        X.loc[X["Glucose"] > 126, "NewGlucose"] = self.NewGlucose[3]
        return X
    
# global variables:
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # to disable caching
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#load models
models = {
    'alzheimer': load_model('./Models/alzheimer_model.keras'),
    'brain_tumor': load_model('./Models/Brain_tumor_model.keras'),
    'breast_cancer': load_model('./Models/breast_cancer_model.pkl'),
    'covid': load_model('./Models/covid_model.h5'),
    'diabetes': load_model('./Models/diabetes_model.pkl'),
    'pneumonia': load_model('./Models/pneumia_model.keras'),
}

model = models['covid19']
# Re-compile the model with the correct settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS # check if the file is an image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<disease>', methods=['GET', 'POST'])
def disease_page(disease):
    prediction_result = None
    if disease in ['alzheimer', 'brain_tumor', 'covid19', 'pneumonia', 'BreastCan', 'diabetes']:
        if request.method == 'POST':  # if the form is submitted
            if disease in ['alzheimer', 'brain_tumor', 'covid19', 'pneumonia']:
                prediction_result = handle_image_disease(request, disease)
            elif disease == 'BreastCan':
                prediction_result = handle_breast_cancer(request)
            elif disease == 'diabetes':
                prediction_result = handle_diabetes(request)
        return render_template(f'{disease}.html', prediction_result=prediction_result)  # render the page with the prediction result
    return redirect(url_for('index'))

def transform_image(file_path,disease,height=150,width=150, color_mode='grayscale'):
    img = load_img(file_path, target_size=(height, width), color_mode=color_mode)
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, axis=0) # add batch dimension
    
    # make the prediction
    model = models[disease]
    prediction = model.predict(img_array)
    result = argmax(prediction,axis=1)[0]
    return result


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

        # Preprocess the image
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

        # Create the categorized folder path
        categorized_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], disease, result)
        if not os.path.exists(categorized_folder_path):
            os.makedirs(categorized_folder_path)

        # Create a unique filename to avoid overwriting
        unique_filename = filename
        counter = 1
        while os.path.exists(os.path.join(categorized_folder_path, unique_filename)):
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1

        # Move the file to the categorized folder
        categorized_file_path = os.path.join(categorized_folder_path, unique_filename)
        os.rename(file_path, categorized_file_path)

        return {'prediction': result, 'file_path': categorized_file_path.replace('\\', '/')}
    return {'error': 'File not allowed'}


def handle_breast_cancer(request):
    features = [
        request.form.get('radius_mean'),
        request.form.get('texture_mean'),
        request.form.get('perimeter_mean'),
        request.form.get('area_mean'),
        request.form.get('smoothness_mean'),
        request.form.get('compactness_mean'),
        request.form.get('concavity_mean'),
        request.form.get('concave_points_mean'),
        request.form.get('symmetry_mean'),
        request.form.get('radius_se'),
        request.form.get('perimeter_se'),
        request.form.get('area_se'),
        request.form.get('compactness_se'),
        request.form.get('concavity_se'),
        request.form.get('concave_points_se'),
        request.form.get('radius_worst'),
        request.form.get('texture_worst'),
        request.form.get('perimeter_worst'),
        request.form.get('area_worst'),
        request.form.get('smoothness_worst'),
        request.form.get('compactness_worst'),
        request.form.get('concavity_worst'),
        request.form.get('concave_points_worst'),
        request.form.get('symmetry_worst'),
        request.form.get('fractal_dimension_worst')
    ]
    
    features = array([float(feature) if feature else nan for feature in features]).reshape(1, -1)
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
        'radius_se', 'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 
        'concave_points_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
        'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    input_data = DataFrame(features, columns=feature_names)

    model = models['breast_cancer']
    prediction = model.predict(input_data)
    breast_cancer_classes = {
        0: 'Benign',
        1: 'Malignant'
    }
    prediction = breast_cancer_classes.get(int(prediction[0]), 'Unknown')
    
    return {'prediction': prediction, 'input_data': input_data.to_dict(orient='records')[0]}

def handle_diabetes(request):
    features = [
        request.form.get('pregnancies'),
        request.form.get('glucose'),
        request.form.get('bloodPressure'),
        request.form.get('skinThickness'),
        request.form.get('insulin'),
        request.form.get('bmi'),
        request.form.get('diabetesPedigreeFunction'),
        request.form.get('age')
    ]
    features = array([float(feature) if feature else nan for feature in features]).reshape(1, -1)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = DataFrame(features, columns=feature_names)

    model = models['diabetes']
    prediction = model.predict(input_data)
    diabetes_classes = {
        0: 'No Diabetes',
        1: 'Diabetes'
    }
    prediction = diabetes_classes.get(int(prediction[0]), 'Unknown')
    return {'prediction': prediction, 'input_data': input_data.to_dict(orient='records')[0]}
if __name__ == '__main__':
    app.run(debug = True,host="0.0.0.0", port= 8000)

