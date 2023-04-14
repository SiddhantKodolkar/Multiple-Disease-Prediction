import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
heart_model = pickle.load(open('heart_disease.pkl', 'rb'))
diabetes_model=pickle.load(open('diabetes.pkl','rb'))
parkinson_model=pickle.load(open('parkinson.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/heart")
def heart():
    return render_template('heart.html')


@app.route("/pred", methods=["GET", "POST"])
def pred():
    age =float(request.form.get("age"))
    sex=float(request.form.get("sex"))
    cp=float(request.form.get("cp"))
    trestbps =float(request.form.get("trestbps"))
    chol =float(request.form.get("chol"))
    fbs =float(request.form.get("fbs"))
    restecg =float(request.form.get("restecg"))
    thalach=float(request.form.get("thalach"))
    exang =float(request.form.get("exang"))
    oldpeak =float(request.form.get("oldpeak"))
    slope =float(request.form.get("slope"))
    ca =float(request.form.get("ca"))
    thal=float(request.form.get("thal"))

    input_tuple = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,
               exang, oldpeak, slope, ca, thal]
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    array_features = pd.DataFrame(input_tuple, columns)
    prediction = heart_model.predict(array_features.T)
    output = prediction[0]

    if output == 1.0:
        return render_template('heart.html', prediction_text="The patient seems to have a heart disease")

    else:
        return render_template('heart.html', prediction_text="The patient does not seem to have a heart disease")


@app.route("/diabetes")
def diabetes():
    return render_template('diabetes.html')


@app.route("/diab", methods=["GET", "POST"])
def diab():
    Pregnancies=float(request.form.get("Pregnancies"))
    Glucose=float(request.form.get("Glucose"))
    BloodPressure=float(request.form.get("BloodPressure"))
    SkinThickness=float(request.form.get("SkinThickness"))
    Insulin=float(request.form.get("Insulin"))
    BMI=float(request.form.get("BMI"))
    DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
    Age=float(request.form.get("Age"))

    input_tuple=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]


    columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


    array_features = pd.DataFrame(input_tuple, columns)
    prediction = diabetes_model.predict(array_features.T)
    output = prediction[0]
    
    if output == 1.0:
        return render_template('diabetes.html', prediction_text="The patient seems to have diabetes")

    else:
        return render_template('diabetes.html', prediction_text="The patient does not seem to have diabetes")


@app.route("/parkinson")
def parkinson():
    return render_template('parkinson.html')

@app.route("/parkin",methods=["POST","GET"])
def parkin():
    input_tuple = [float(i) for i in request.form.values()]
    columns=['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']

    array_features = pd.DataFrame(input_tuple, columns)
    prediction = parkinson_model.predict(array_features.T)
    output = prediction[0]
    
    if output == 1.0:
        return render_template('parkinson.html', prediction_text="The patient seems to have Parkison's Disease")

    else:
        return render_template('parkinson.html', prediction_text="The patient does not seem to have Parkison's Disease")

if __name__ == '__main__':
    app.run(debug=True)
