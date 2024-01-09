from flask import Flask,render_template
from flask import request
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict_disease():
    age=int(request.form.get('age'))
    gender=int(request.form.get('gender'))
    chest=int(request.form.get('chest'))
    rbp=int(request.form.get('rbp'))
    cholestrol=int(request.form.get('cholestrol'))
    fbs=int(request.form.get('fbs'))
    rest_ecg_result=int(request.form.get('rest-ecg-result'))
    mhr=int(request.form.get('mhr'))
    eia=int(request.form.get('eia'))
    st_depression=float(request.form.get('st-depression'))
    st=int(request.form.get('st'))
    no_of_vessels=int(request.form.get('no-of-vessels'))
    thallium=int(request.form.get('thallium'))
    input_data=np.array([age,gender,chest,rbp,cholestrol,fbs,rest_ecg_result,mhr,eia,st_depression,st,no_of_vessels,thallium]).reshape(1,13)
    input_data=scaler.transform(input_data)
    result=model.predict(input_data)
    if(result[0]==1):
        result='Present'
    else:
        result='Absent'
    result='Heart Disease is ' + result
    return render_template('index.html',result=result)
if __name__=='__main__':
    app.run(debug=False,host="0.0.0.0")