from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
from utils.errors import InvalidUsage
from config.model_config import ModelConfig
from models.train import ClassificationTrainer
from models.utils import DataPreporcess
from models.inference import ClassificationInference
import pandas as pd
import torch
import os

class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GenrePredictor(Resource, metaclass=Singleton):
    model = None
    
    def __init__(self):
        if not self.model:
            self.model = ClassificationInference(ModelConfig)

    def classify_genre(self, overview):
        return self.model.get_predictions(overview)
    
    def post(self):
        try:
            # fetech overview from request 
            overview = request.form['overview']

            # Predict the genre
            genre = self.classify_genre(overview)
            
            return {'genre': tuple(genre)}, 200
        
        except Exception as e:
            raise InvalidUsage('Wrong', status_code=500)

np.random.seed(ModelConfig.SEED)
torch.manual_seed(ModelConfig.SEED)

def train_model():
    if os.path.exists(ModelConfig.MODEL_LOC):
        df = pd.read_pickle(ModelConfig.EMBEDDING_DATA_LOC)
    else:
        df = DataPreporcess(ModelConfig).get_training_data()
    ClassificationTrainer(ModelConfig).split_data_and_train(df)


def create_app():
    if os.path.exists(ModelConfig.MODEL_LOC) and os.path.exists(ModelConfig.ENCODER_LOC):
            pass
    else:
        print('Training Model')
        train_model()
    
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(GenrePredictor, '/')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(port=8000, debug=True)
