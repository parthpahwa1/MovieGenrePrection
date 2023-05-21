import torch
from sklearn.preprocessing import LabelEncoder
from .utils import MLP, DataPreporcess
import numpy as np

class ClassificationInference(torch.nn.Module):
    """
    A PyTorch module that wraps a classification model for inference.

    This module loads a pre-trained model and an associated label encoder, preprocesses input data,
    performs inference, and then transforms model predictions back into original labels.

    Args:
        ModelConfig (Config): Configuration object containing parameters for data preprocessing,
        the model, and paths for loading the pre-trained model and the label encoder.
    """

    config = None
    model = None
    data_processor = None

    def __init__(
            self,
            ModelConfig,
        ):
        super(ClassificationInference, self).__init__()
        self.config = ModelConfig

        # Initialize a data preprocessor using the configuration
        self.data_processor = DataPreporcess(ModelConfig)

        # Initialize the model using parameters from the configuration
        self.model = MLP(
            num_categories = self.config.NUM_CATEGORIES,
            input_size = self.config.INPUT_SIZE,
            hidden_size = self.config.HIDDEN_DIM,
            num_layers = self.config.NUM_HIDDEN_LAYER,
            dropout = self.config.DROPOUT
        )

        # Load the pre-trained weights into the model and set it to evaluation mode
        self.model.load_state_dict(torch.load(self.config.MODEL_LOC))
        self.model.eval()

        # Initialize a label encoder and load its state
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(self.config.ENCODER_LOC)

    def get_predictions(self, input_, threshold=0.3):
        """
        Perform inference on the input and return labels exceeding the threshold.

        Args:
            input_ (str): The input text to classify.
            threshold (float, optional): The threshold for selecting labels based on their
            predicted probabilities.

        Returns:
            np.array: The predicted labels.
        """
        # Get model predictions
        preds = self(input_)
        preds = preds.sigmoid().detach().numpy()

        # Get the labels that have a predicted probability higher than the threshold
        label_list = np.where(preds > threshold)[1]

        # If no labels exceed the threshold, select the label with the highest probability
        if len(label_list) == 0:
            label_list = np.argmax(preds)
        
        return self.encoder.inverse_transform(label_list)

    def forward(self, input_):
        """
        The forward pass of the model.

        Args:
            input_ (str): The input text to classify.

        Returns:
            torch.Tensor: The model's raw output (logits).
        """
        # Preprocess the input data
        embd = self.data_processor.sentence_transformer.encode([input_])
        print(self.data_processor)
        # Perform inference and return the model's raw output (logits)
        return self.model(torch.FloatTensor(embd))