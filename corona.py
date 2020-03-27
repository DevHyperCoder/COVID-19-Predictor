import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn
import pickle
from flask import render_template, Flask, redirect, request
import os
from waitress import serve

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app,listen = "*:"+port)

df = pd.read_csv("covid.csv", names=['age',
                                     'chronic',
                                     'travelled',
                                     'runny',
                                     'fever',
                                     'diff_breath',
                                     'predict'
                                     ])

print(df)

x = np.array(df.drop(['predict'], 1))
y = np.array(df['predict'])

print(x.shape)
print(y.shape)

# x=np.array(x).reshape(1,-1)
# y=np.array(y).reshape(1,-1)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=1000)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

forest.fit(x_train, y_train)

print(forest.predict(np.array([80, 1, 1, 1, 1, 1]).reshape(1, -1)))

print(forest.predict(np.array([80, 0, 1, 0, 0, 0]).reshape(1, -1)))
#
# print(forest.predict(x_test))
from sklearn.model_selection import cross_val_score
# print(cross_val_score(forest,x,y,cv=3))
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

best = 0
# for i in range(10):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)
#
#     forest.fit(x_train, y_train)
#     acc = forest.score(x_test, y_test)
#     print(acc)
#     if acc > best:
#         best = acc
#         with open("student.pickle", "wb") as f:
#             pickle.dump(forest, f)

# forest = pickle.load(open("student.pickle","rb"))
forest.fit(x_train,y_train)
print(forest.score(x_test,y_test))
print(forest.predict(np.array([34, 0, 1, 0, 0, 0]).reshape(1, -1)))

# print(forest.predict(np.array[34,0,0,0,0,0].reshape(1,-1)))




@app.route("/predict" ,methods = ['GET','POST'])
def index():

    age = request.form["age"]
    chronic = request.form["chronic"]
    travelled = request.form["travelled"]
    runny = request.form["runny"]
    fever = request.form["fever"]
    diff_breath = request.form["diff_breath"]

    if chronic == "YES":
        chronic = 1
    else:
        chronic =0

    if travelled == "YES":
        travelled = 1
    else:
        travelled =0

    if runny == "YES":
        runny = 1
    else:
        runny =0

    if fever == "YES":
        fever = 1
    else:
        fever =0

    if diff_breath == "YES":
        diff_breath = 1
    else:
        diff_breath =0




    predict=forest.predict(np.array([age,chronic,travelled,runny,fever,diff_breath]).reshape(1,-1))

    if(chronic == 1 and travelled == 1 and runny == 1 and fever == 1 and diff_breath == 1 and age > "79"):
        if predict is not 1:
            predict = 1

    return render_template("display.html",predict=predict)

#     return render_template("display.html",predict=predict)

# if __name__ == "corona":
#     serve(app,listen = "*:8080")
