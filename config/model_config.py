from dataclasses import dataclass

@dataclass
class ModelConfig():
    SEED: float = 42
    DROPOUT: float = 0.2
    NUM_HIDDEN_LAYER: float = 3
    HIDDEN_DIM: float = 256
    LR: float = 1e-4
    EPOCHS: int = 25
    SCHEDULDER_GAMMA: float = 0.97
    BATCH_SIZE: int = 128
    INPUT_COLUMN: str = 'overview'
    TARGET_COLUMN: str = 'genres'
    PRE_TRAINED_MODEL: str = 'all-MiniLM-L6-v2'
    MODEL_LOC: str = './models/artifacts/mlp.pkl'
    VALIDATION_RESULTS_LOC: str = './models/artifacts/perfromance.csv'
    TRAIN_DATA_LOC: str = './data/movies_metadata.csv'
    ENCODER_LOC: str = './models/artifacts/encoded_classes.npy'
    EMBEDDING_DATA_LOC: str = './data/data_with_embedding.pkl'
    NUM_CATEGORIES: int = 33
    INPUT_SIZE: int = 384
