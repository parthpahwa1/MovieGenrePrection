from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
from app.utils.errors import InvalidUsage
from app.config.model_config import ModelConfig
from app.models.train import ClassificationTrainer
from app.models.utils import DataPreporcess
from app.models.inference import ClassificationInference
import pandas as pd
import torch
import os

class Singleton(type):
    """
    A Singleton metaclass. Any class using Singleton as its metaclass will be a singleton class.
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        """
        Overrides the call method for creating objects. If an instance of a class with Singleton as
        its metaclass already exists, this instance is returned, otherwise a new instance is created
        and stored for future use.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GenrePredictor(Resource, metaclass=Singleton):
    """
    A GenrePredictor class responsible for handling POST requests and classifying the genre of a text. 
    Inherits from Resource and uses Singleton as its metaclass.
    """
    model = None
    
    def __init__(self):
        """
        Initialize the GenrePredictor. If the model has not been loaded, it's done here.
        """
        if not self.model:
            self.model = ClassificationInference(ModelConfig)

    def classify_genre(self, overview):
        """
        Classify the genre of a given text.

        Args:
            overview (str): The overview text.

        Returns:
            list: The predicted genre labels.
        """
        return self.model.get_predictions(overview)
    
    def post(self):
        """
        Handle POST requests, get the overview text from the request, classify the genre and return it.

        Returns:
            tuple: Tuple containing a dictionary of genres and a status code.
        """
        try:
            # fetech overview from request 
            overview = request.form['overview']

            # Predict the genre
            genre = self.classify_genre(overview)
            
            return {'genre': tuple(genre)}, 200
        
        except Exception as e:
            raise InvalidUsage('Wrong', status_code=500)

def train_model():
    """
    Train the genre prediction model if it has not been trained yet.
    """
    if os.path.exists(ModelConfig.MODEL_LOC):
        df = pd.read_pickle(ModelConfig.EMBEDDING_DATA_LOC)
    else:
        df = DataPreporcess(ModelConfig).get_training_data()
    ClassificationTrainer(ModelConfig).split_data_and_train(df)


def create_app():
    """
    Create the Flask app, train the model if necessary, and add the GenrePredictor resource.

    Returns:
        app: The created Flask app.
    """
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
    np.random.seed(ModelConfig.SEED)
    torch.manual_seed(ModelConfig.SEED)
    app = create_app()
    app.run(port=8000, debug=True)
