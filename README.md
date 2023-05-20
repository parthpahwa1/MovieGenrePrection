### Methodology
We use SentenceTransfomers to create embeddings for the movie overview and train a shallow NN to predict the genre.
Given a movie may have multiple genres, we treat this as a multi lable classificaiton problem.

Step 1. Create embeddings for each review

Step 2. Create multilabel targets of the format [1, 0, 1, 0, 0 ...] where 0 dentoes genre not present

Step 3. Train an MLP with BCEWithLogitsLoss

### Setup 

Note this code uses poetry for package management - To install poetry please follow https://python-poetry.org/docs/#installing-with-the-official-installer

``poetry install``

### Run

``poetry run python run.py``

### Structure
> config
* Contains all constant and config values

> data
* movies_metadata.csv - raw csv file 
* data_with_embedding.pkl - cached sentence embedding and encoded target data

> models
* train.py : Trainer code, accepts TensorDataset as input, trains and saves the model artifact
* inference.py : Inference code

