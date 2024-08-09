from sklearn.base import BaseEstimator, TransformerMixin
from pandas import Series, to_numeric

# TODO:Creating Custom Transformers
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
    