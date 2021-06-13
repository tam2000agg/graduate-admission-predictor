from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd
import numpy as np
#import pickle5 as pickle

# ML Pkg
# from sklearn.externals import joblib
import joblib 
from joblib import load,dump

app = Flask(__name__,static_url_path='/static')
Material(app)


@app.route("/")
def index():
	return render_template('index.html')


@app.route("/preview")
def preview():
	print("preview")
	df = pd.read_csv("data/Admission_Predict.csv")
	return render_template("preview.html",df_view = df)


@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		gre_score = request.form['gre_score']
		toefl_score = request.form['toefl_score']
		university = request.form['university']
		sop_value = request.form['sop_value']
		lor_value = request.form['lor_value']
		cgpa_value = request.form['cgpa_value']
		research = request.form['research']
		model_choice = request.form['model_choice']
		print("GDJD")
		print(research)
		# Clean the data by convert from unicode to float
		print(type(gre_score))

		sample_data = np.array([gre_score,toefl_score,university,sop_value,lor_value,cgpa_value,research])
		print(type(sample_data))
		clean_data =[float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1, -1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		print(model_choice)
		if model_choice == 'logitmodel':
			print("model_choice")
			logit_model = joblib.load('data/LR_model.pkl')
			result_prediction = logit_model.predict(ex1)
			minimum = 7.86099772
			maximum = 9.56157506
			diff = (maximum-minimum)*(0.1)
			if result_prediction <= minimum + diff :
				result_prediction = 1
				print(result_prediction)
			elif result_prediction <= minimum + (2*diff) :
				result_prediction = 2
				print(result_prediction)
			elif result_prediction <= minimum + (3*diff) :
				result_prediction = 3
				print(result_prediction)
			elif result_prediction <= minimum + (4*diff) :
				result_prediction = 4
				print(result_prediction)
			elif result_prediction <= minimum + (5*diff) :
				result_prediction = 5
				print(result_prediction)
			elif result_prediction <= minimum + (6*diff) :
				result_prediction = 6
				print(result_prediction)
			elif result_prediction <= minimum + (7*diff) :
				result_prediction = 7
				print(result_prediction)
			elif result_prediction <= minimum + (8*diff) :
				result_prediction = 8
				print(result_prediction)
			elif result_prediction <= minimum + (9*diff) :
				result_prediction = 9
				print(result_prediction)
			else :
				result_prediction =10
				print(result_prediction)

		else :
			pass


	return render_template('index.html', gre_score=gre_score,
		toefl_score=toefl_score,
		university=university,
		sop_value=sop_value,
		lor_value=lor_value,
		cgpa_value=cgpa_value,
		research=research,
		result_prediction=result_prediction,
		clean_data=clean_data,
		model_selected=model_choice)

if __name__ == '__main__':
	app.run(debug=True)
