from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
import numpy as np
import pandas as pd
import os

def predict_genres(movie):
    
    model = joblib.load(os.path.dirname(__file__) + '\\genremovies.pkl') 
    vect = joblib.load(os.path.dirname(__file__) + '\\vectorizer_tfid.pkl')
    col_p = ['Action','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama','Family','Fantasy','Film-Noir','History','Horror','Music','Musical','Mystery','News','Romance','Sci-Fi','Short','Sport','Thriller','War','Western']
    
    if movie != 'https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip':
        entry = vect.transform([movie])
        pred = model.predict_proba(entry)
        
        index = pred.argsort()[0][-3:][::-1]
        probas = np.around(pred[0][index]*100,2)
        
        respuesta = f"los generos mas probables son: {col_p[index[0]]} con una probabilidad de {probas[0]}%"
        respuesta += f", {col_p[index[1]]} con una probabilidad de {probas[1]}%"
        respuesta += f", {col_p[index[2]]} con una probabilidad de {probas[2]}%."
        
        return respuesta
    else: 
        respuesta = "No hay data"
        return respuesta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='API Prediccion generos de peliculas',
    description='API que al ingresar la descripcion de una pelicula (en ingles) predice el genero de pelicula mas probable.')

ns = api.namespace('predict', 
     description='Clasificador de generos')
   
parser = api.parser()

parser.add_argument(
    'Descripcion', 
    type=str, 
    required=True, 
    help='Descripcion de pelicula, solo texto (en ingles). Sin comillas.', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_genres(args['Descripcion'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)