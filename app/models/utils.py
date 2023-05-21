import numpy as np
import os
import torch
import pandas as pd
import torch.nn.functional as F
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

def one_hot(n_classes, targets):
    """
    One-hot encode targets.

    Args:
        n_classes (int): The total number of classes.
        targets: The targets to be encoded.

    Returns:
        numpy.array: The one-hot encoded targets.
    """
    one_hot_traget = np.eye(n_classes)[targets]
    return one_hot_traget

def encode_target(row, encoder):
    """
    Encode the target row.

    Args:
        row: The row of targets to be encoded.
        encoder (LabelEncoder): An instance of sklearn's LabelEncoder.

    Returns:
        numpy.array: The encoded targets.
    """
    num_classes = len(encoder.classes_)

    target = np.zeros((1, num_classes))

    for label in row:
        target += one_hot(num_classes, encoder.transform([label]))

    return np.clip(target, 0, 1)

def get_genres_for_row(row):
    """
    Extract genre names from the row.

    Args:
        row: The row of data containing genre information.

    Returns:
        list: A list of genre names.
    """
    row = literal_eval(row)
    result = []

    for element in row:
        result.append(element['name'])

    return result


class MLP(torch.nn.Module):
    """
    A PyTorch module for a multi-layer perceptron (MLP).

    Args:
        input_size (int): The size of the input layer.
        num_categories (int): The size of the output layer.
        hidden_size (int): The size of the hidden layers.
        num_layers (int): The number of hidden layers.
        dropout (float): The dropout rate.
    """
    def __init__(
            self,
            input_size: int,
            num_categories: int,
            hidden_size: int,
            num_layers: int,
            dropout: float
    ):
        super(MLP, self).__init__()

        self.output_size = num_categories
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.create_model()

    def create_model(self):
        current_dim = self.input_size
        
        for _ in range(self.num_layers):
            self.layers.append(torch.nn.Linear(current_dim, self.hidden_size))
            current_dim = self.hidden_size
        
        self.layers.append(torch.nn.Linear(current_dim, self.output_size))

    def forward(self, x):
        x = F.relu(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = self.dropout_layer(F.relu(layer(x) + x))
        
        out = self.layers[-1](x)
        return out

class DataPreporcess():
    """
    A class for data preprocessing.

    Args:
        ModelConfig (Config): Configuration object containing parameters for the model and data preprocessing.
    """
    def __init__(self, ModelConfig):
        self.config = ModelConfig

        self.sentence_transformer = SentenceTransformer(self.config.PRE_TRAINED_MODEL)
    
    def create_encoder(self, categories):
        """
        Create a label encoder and save the classes.

        Args:
            categories (list): A list of unique categories.

        Returns:
            LabelEncoder: An instance of sklearn's LabelEncoder.
        """
        encoder = LabelEncoder()
        encoder.fit(categories)

        np.save(self.config.ENCODER_LOC, encoder.classes_)
        return encoder

    def get_targets(self, df:pd.DataFrame):
        """
        Get the encoded targets from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing target column.

        Returns:
            pd.Series: A Series of encoded targets.
        """
        if os.path.exists(self.config.ENCODER_LOC):
            encoder = LabelEncoder()
            encoder.classes_ = np.load(self.config.ENCODER_LOC)
        
        else:
            categories = list(set(df[self.config.TARGET_COLUMN].explode().tolist()))
            encoder = self.create_encoder(categories)

        return df[self.config.TARGET_COLUMN].apply(lambda row: encode_target(row, encoder))

    def preocess_training_data(self, df):
        """
        Preprocess the training data.

        Args:
            df (pd.DataFrame): The DataFrame containing the raw training data.

        Returns:
            pd.DataFrame: The DataFrame containing the processed training data.
        """
        embeddings = self.get_embedding(df[self.config.INPUT_COLUMN])
        df['x'] = None
        df['x'] = [embeddings[i, :] for i in range(0, len(embeddings))]

        df[self.config.TARGET_COLUMN] = df[self.config.TARGET_COLUMN].apply(lambda row: get_genres_for_row(row))
        df['y'] = None

        df['y'] = self.get_targets(df)
        return df[['x', 'y']]
        

    def get_training_data(self):
        data_loc = self.config.TRAIN_DATA_LOC
        requried_columns = [self.config.INPUT_COLUMN, self.config.TARGET_COLUMN]

        if not set(requried_columns).issubset(pd.read_csv(data_loc, nrows=1).columns):
            raise ValueError(f"The input CSV does not contain required columns: {requried_columns}")
        
        df = pd.read_csv(data_loc)[requried_columns].dropna()
        df = df[df[self.config.INPUT_COLUMN].str.len() > 5].reset_index(drop=True)

        df = self.preocess_training_data(df[requried_columns])

        df.to_pickle(self.config.EMBEDDING_DATA_LOC)
        return df

    def get_embedding(self, text):
        """
        Get the sentence embeddings for the given text.

        Args:
            text (str): The input text.

        Returns:
            numpy.array: The sentence embeddings.
        """
        return self.sentence_transformer.encode(text)
