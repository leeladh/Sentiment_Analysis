from flask import Flask, redirect, request, url_for, render_template, flash
from model import *

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def hello():
	if request.method=="POST":
		sentiment = request.form.get('sentiment')
		model = request.form.get('model')
		res = get_result(model, sentiment)
		return render_template('home.html', analysis=res)
	return render_template('home.html')

if __name__ == "__main__":
	app.run(debug=True)