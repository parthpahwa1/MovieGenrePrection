import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import MLP
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

class ClassificationTrainer(torch.nn.Module):
    """
    A PyTorch module for training a classification model.

    Args:
        ModelConfig (Config): Configuration object containing parameters for the model, optimizer, 
                              scheduler and various training and validation aspects.
    """
    def __init__(
            self,
            ModelConfig
        ):
        super(ClassificationTrainer, self).__init__()
        self.config = ModelConfig

        # Initialize the model using parameters from the configuration
        self.model = MLP(
            num_categories = self.config.NUM_CATEGORIES,
            input_size = self.config.INPUT_SIZE,
            hidden_size = self.config.HIDDEN_DIM,
            num_layers = self.config.NUM_HIDDEN_LAYER,
            dropout = self.config.DROPOUT
        )

        # Initialize the loss function
        self.loss = torch.nn.BCEWithLogitsLoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            params = self.model.parameters(),
            lr = self.config.LR
        )

        # Initialize the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.SCHEDULDER_GAMMA)

    def split_data_and_train(self, df):
        """
        Split the data into train and test set, and then train the model.

        Args:
            df (DataFrame): The input dataframe containing 'x' and 'y' columns.

        Raises:
            KeyError: If 'x' or 'y' column is not found in the dataframe.
        """

        if not {'x', 'y'}.issubset(df.columns):
            raise KeyError("'x' and 'y' columns must be present in the dataframe.")

        x_train, x_test, y_train, y_test =  train_test_split(df['x'], df['y'], test_size=0.2, random_state=self.config.SEED)

        x_train = torch.FloatTensor(np.vstack(x_train))
        x_test = torch.FloatTensor(np.vstack(x_test))

        y_train = torch.FloatTensor(np.vstack(y_train))
        y_test = torch.FloatTensor(np.vstack(y_test))

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_test, y_test)

        self.train(train_dataset, val_dataset)


    def train(self, train_data, val_data=None):
        """
        Train the model.

        Args:
            train_data (TensorDataset): The training data.
            val_data (TensorDataset, optional): The validation data. If None, no validation is performed.
        """
        train_data = DataLoader(train_data, batch_size=self.config.BATCH_SIZE)

        # Iterate over epochs
        for epoch in range(self.config.EPOCHS):
            total_loss = 0
            validation_loss = 0
            train_loss = 0

            # Train on batches
            for indx, batch in tqdm(enumerate(train_data)):
                _, loss = self._train_step(batch[0], batch[1])
                total_loss += loss
            train_loss = total_loss/len(train_data)

            # Validate the model
            if val_data is not None:
                val = DataLoader(val_data, batch_size=self.config.BATCH_SIZE)
                total_loss = 0

                for indx, batch in enumerate(val):
                    loss = self._loss(self(batch[0]), batch[1])
                    total_loss += loss
                
                validation_loss = total_loss/len(val)

            print('Epoch: {}, train loss: {}, validation loss:{}'.format(epoch, train_loss, validation_loss))
            self.scheduler.step()

        # Evaluate on the validation data and print report if available
        if val_data is not None:
            self.evaluate(val_data)

        torch.save(
            self.model.state_dict(),
            self.config.MODEL_LOC
        )

    def evaluate(self, val_data):
        """
        Evaluate the model on validation data and print classification report.

        Args:
            val_data (TensorDataset): The validation data.
        """
        val = DataLoader(val_data, batch_size=self.config.BATCH_SIZE)
        pred_list = []
        true_list = []
        threshold = 0.3
        for indx, batch in enumerate(val):
            preds = self(batch[0])
            preds = preds.sigmoid().detach().numpy()

            sol = list((preds>threshold).astype(int))
            
            true_list += list((batch[1].detach().numpy().astype(int)))
            pred_list += sol

        report = classification_report(
            true_list, 
            pred_list, 
            output_dict=True, 
            target_names=np.load(self.config.ENCODER_LOC),
            zero_division=0
        )
        report = pd.DataFrame(report).transpose()
        print(report)
        report.to_csv(self.config.VALIDATION_RESULTS_LOC)


    def _loss(self, x, y):
        """
        Compute the loss on given inputs and targets.

        Args:
            x (torch.Tensor): The inputs.
            y (torch.Tensor): The targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.loss(x, y)

    def _train_step(self, x, y):
        """
        Perform a single training step (forward pass, loss computation and backward pass).

        Args:
            x (torch.Tensor): The inputs.
            y (torch.Tensor): The targets.

        Returns:
            tuple: The model outputs and the computed loss.
        """
        pred = self(x)
        loss = self._loss(pred, y)
        self._update_weights(loss)
        return pred, loss

    def _update_weights(self, loss):
        """
        Perform the backward pass and update the model weights.

        Args:
            loss (torch.Tensor): The computed loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.model(x)