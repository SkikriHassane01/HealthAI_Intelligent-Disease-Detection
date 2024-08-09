# TODO:______________________________________________________importing the libraries______________________________________________________

from flask import Flask, request, redirect, url_for, render_template
from numpy import array,nan
from pandas import DataFrame
import os 
import joblib, pickle
from werkzeug.utils import secure_filename 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.base import BaseEstimator, TransformerMixin

# TODO:______________________________________________________creating and configuring the flask app______________________________________________________
