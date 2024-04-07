from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)


with open('modele.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/')
def  home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def result():
    # Récupérez les valeurs saisies par l'utilisateur à partir du formulaire
    features = [float(x) for x in request.form.values()]

    # Prédiction du prix à partir des caractéristiques saisies
    predicted_price = model.predict([features])

    return render_template('predict.html', price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)