from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == "POST":

		with open('model/knn.pkl', 'rb') as r:
			model = pickle.load(r)

		umur = float(request.form['umur'])
		berat = float(request.form['berat'])
		tinggi = float(request.form['tinggi'])

		
		datas = np.array((umur, berat, tinggi))
		datas = np.reshape(datas, (1, -1))

		isKlasifikasi = model.predict(datas)

		return render_template('hasil.html', finalData=isKlasifikasi)

	else:
		return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)