import torch
from sklearn.preprocessing import LabelEncoder
from .utils import MLP, DataPreporcess
import numpy as np

class ClassificationInference(torch.nn.Module):
    def __init__(
            self,
            ModelConfig,
        ):
        super(ClassificationInference, self).__init__()
        self.config = ModelConfig

        self.data_processor = DataPreporcess(ModelConfig)
        self.model = MLP(
            num_categories = self.config.NUM_CATEGORIES,
            input_size = self.config.INPUT_SIZE,
            hidden_size = self.config.HIDDEN_DIM,
            num_layers = self.config.NUM_HIDDEN_LAYER,
            dropout = self.config.DROPOUT
        )

        self.model.load_state_dict(torch.load(self.config.MODEL_LOC))
        self.model.eval()

        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(self.config.ENCODER_LOC)

    def get_predictions(self, input_, threshold=0.3):
        preds = self(input_)
        preds = preds.sigmoid().detach().numpy()
        label_list = np.where(preds > threshold)[1]

        if len(label_list) == 0:
            label_list = np.argmax(preds)
        
        return self.encoder.inverse_transform(label_list)

    def forward(self, input_):
        embd = self.data_processor.sentence_transformer.encode([input_])
        return self.model(torch.FloatTensor(embd))