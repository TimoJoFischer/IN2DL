"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
import matplotlib.pyplot as plt

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):

        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        self.model = nn.Sequential(
            nn.Conv2d(1,7,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(7, 7, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(7, 7, 11, padding=5),
            nn.ReLU(),
            nn.Conv2d(7, 7, 15, padding=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(7*24*24,50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50,30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        x = x.view(x.shape[0],1,96,96)
        x = self.model(x.float())
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        images,targets = batch['image'],batch['keypoints']
        targets = targets.reshape(targets.shape[0],-1)
        out = self.forward(images)
        loss = torch.nn.functional.l1_loss(out,targets)
        if batch_idx==0 and mode=="val":
            indexes=np.random.choice(targets.shape[0], 5, replace=False)
            self.visualize_predictions(images[indexes],targets[indexes],out[indexes])
        return loss

    def training_step(self,batch,batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'train_loss': loss}
        return {'loss':loss,'log': tensorboard_logs}

    def validation_step(self,batch,batch_idx):
        loss= self.general_step(batch,batch_idx,"val")
        return {'val_loss':loss}

    def validation_end(self, outputs):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self,batch,batch_idx):
        loss = self.general_step(batch,batch_idx,"test")
        return {'test_loss':loss}

    def configure_optimizers(self):
        optim= torch.optim.Adam(self.parameters(),lr=self.hparams['lr'])
        return optim

    def prepare_data(self):
        FACIAL_ROOT = "../datasets/facial_keypoints"
        #Todo transform train set
        my_transform=None
        self.dataset={}
        self.dataset['train']=FacialKeypointsDataset(root=FACIAL_ROOT,train=True,transform=my_transform)
        self.visualize_transformed_images()
        self.dataset['val']=FacialKeypointsDataset(root=FACIAL_ROOT,train=False,transform=None)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    def visualize_transformed_images(self):
        fig = plt.figure(figsize=(15, 5))
        for i in range(10):
            image_target=self.dataset['train'][i]
            plt.subplot(2, 5, i + 1)
            plt.imshow(image_target['image'].reshape(96,96))
            plt.scatter(image_target['keypoints'][:, 0]*48+48, image_target['keypoints'][:, 1]*48+48, s=100, marker='.', c='g')
        self.logger.experiment.add_figure("transformations",fig,global_step=self.global_step)

    def visualize_predictions(self,images,labels,predictions):
        fig = plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i].reshape(96, 96))
            label = labels[i].reshape(-1, 2).detach().numpy() * 48 + 48
            prediction = predictions[i].reshape(-1, 2).detach().numpy() * 48 + 48
            plt.scatter(label[:, 0], label[:, 1], s=100, marker='.', c='g')
            plt.scatter(prediction[:, 0], prediction[:, 1], s=100, marker='.', c='r')
        self.logger.experiment.add_figure("predictions", fig, global_step=self.global_step)


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)


