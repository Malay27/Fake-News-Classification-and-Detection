from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/',methods=['POST'])
def predict():
	filename='model_MNB.pkl'
	model_MNB = pickle.load(open(filename, 'rb'))

	filename='model_DT.pkl'
	model_DT = pickle.load(open(filename, 'rb'))

	filename='model_RF.pkl'
	model_RF = pickle.load(open(filename, 'rb'))

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction_MNB = model_MNB.predict(data)
		my_prediction_DT = model_DT.predict(data)
		my_prediction_RF = model_RF.predict(data)
	
	return render_template('result.html', prediction_MNB = my_prediction_MNB,prediction_DT = my_prediction_DT,prediction_RF = my_prediction_RF)

if __name__ == '__main__':
	app.run(debug=True)