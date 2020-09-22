import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
RF_Selector = pickle.load(open('RF_Model.pkl', 'rb'))
RF_Selector = joblib.load('RF_Model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        Elevation = request.form["Elevation"]
        Horizontal_Distance_To_Roadways = float(request.form["Horizontal_Distance_To_Roadways"])
        soil_type = float(request.form["soil_type"])
        Road_fire = float(request.form["Road_fire"])
        EL_DIS = float(request.form["EL_DIS"])
        EL_Fire = float(request.form["EL_Fire"])        
        EL_Road = float(request.form["EL_Road"])
        
        
        prediction=RF_Selector.predict([[
            Elevation,
            Horizontal_Distance_To_Roadways,
            soil_type,
            Road_fire,
            EL_DIS,
            EL_Fire,
            EL_Road
        ]])

        output=round(prediction[0],2)

        return render_template('index.html',prediction_text="Forest Coverage Classified as Type {}".format(int(output)))


    return render_template("index.html")






if __name__ == "__main__":
    app.run(debug=True)
