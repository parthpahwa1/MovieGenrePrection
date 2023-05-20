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
    def __init__(
            self,
            ModelConfig
        ):
        super(ClassificationTrainer, self).__init__()
        self.config = ModelConfig


        self.model = MLP(
            num_categories = self.config.NUM_CATEGORIES,
            input_size = self.config.INPUT_SIZE,
            hidden_size = self.config.HIDDEN_DIM,
            num_layers = self.config.NUM_HIDDEN_LAYER,
            dropout = self.config.DROPOUT
        )

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            params = self.model.parameters(),
            lr = self.config.LR
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.SCHEDULDER_GAMMA)

    def split_data_and_train(self, df):
        x_train, x_test, y_train, y_test =  train_test_split(df['x'], df['y'], test_size=0.2, random_state=self.config.SEED)

        x_train = torch.FloatTensor(np.vstack(x_train))
        x_test = torch.FloatTensor(np.vstack(x_test))

        y_train = torch.FloatTensor(np.vstack(y_train))
        y_test = torch.FloatTensor(np.vstack(y_test))

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_test, y_test)

        self.train(train_dataset, val_dataset)


    def train(self, train_data, val_data=None):
        train_data = DataLoader(train_data, batch_size=self.config.BATCH_SIZE)

        for epoch in range(self.config.EPOCHS):
            total_loss = 0
            validation_loss = 0
            train_loss = 0

            for indx, batch in tqdm(enumerate(train_data)):
                _, loss = self._train_step(batch[0], batch[1])
                total_loss += loss
            train_loss = total_loss/len(train_data)

            if val_data is not None:
                val = DataLoader(val_data, batch_size=self.config.BATCH_SIZE)
                total_loss = 0

                for indx, batch in enumerate(val):
                    loss = self._loss(self(batch[0]), batch[1])
                    total_loss += loss
                
                validation_loss = total_loss/len(val)

            print('Epoch: {}, train loss: {}, validation loss:{}'.format(epoch, train_loss, validation_loss))
            self.scheduler.step()

        if val_data is not None:
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
                target_names=np.load(self.config.ENCODER_LOC)
            )
            report = pd.DataFrame(report).transpose()
            print(report)
            report.to_csv(self.config.VALIDATION_RESULTS_LOC)


        torch.save(
            self.model.state_dict(),
            self.config.MODEL_LOC
        )


    def _loss(self, x, y):
        return self.loss(x, y)

    def _train_step(self, x, y):
        pred = self(x)
        loss = self._loss(pred, y)
        self._update_weights(loss)
        return pred, loss

    def _update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.model(x)