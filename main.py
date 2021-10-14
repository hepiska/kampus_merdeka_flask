from flask import Flask
from flask import render_template
from flask import request
import numpy as np
import pickle


app = Flask(__name__)

iris_model_file = open("models/iris_petal_model.pkl", "rb")

iris_model = pickle.load(iris_model_file)

@app.route("/")
def alive():
  return "alive !!"

@app.route("/test")
def test():
  myDict = {
    "name": "ego"
  }
  return myDict

@app.route("/hello/<name>")
def hello(name):

  return render_template("hello.html", name = name)

@app.route("/form")
def form():
  
  return render_template("predict.html")

@app.route("/predict_res", methods=['POST'])
def predict_res():
  iris_target_name = ['setosa', 'versicolor', 'virginica']
  data_array = [float(x) for x in request.form.values()]
  data = [np.array(data_array)]
  result = iris_model.predict(data)
  print("predict", iris_target_name[result[0]])
  return render_template("predict_res.html", iris_name = iris_target_name[result[0]]) 