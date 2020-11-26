import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from exercise_code.data.segmentation_dataset import SegmentationData


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.model = nn.Sequential(
            nn.Conv2d(3, 9, 3),
            nn.ReLU(),
            nn.Conv2d(9, 37, 3),
            nn.ReLU(),
            nn.Conv2d(37, 111, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(111, 80, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 70, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(70, 60, 3),
            nn.ReLU(),
            nn.Conv2d(60,40,1),
            nn.ReLU(),
            nn.Conv2d(40,23,1)
        )

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.model(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def general_step(self, batch, batch_idx, type):
        images, targets = batch
        outs = self.forward(images)
        loss = self.loss_func(outs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self,batch,batch_idx):
        loss= self.general_step(batch,batch_idx,"val")
        return {'val_loss':loss}

    def validation_end(self, outputs):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optim

    def prepare_data(self):
        SEGMENTATION_ROOT = "../datasets/segmentation/segmentation_data/"
        self.dataset = {}
        self.dataset['train'] = SegmentationData(image_paths_file=SEGMENTATION_ROOT + "train.txt")
        self.dataset['val'] = SegmentationData(image_paths_file=SEGMENTATION_ROOT + "val.txt")
        self.dataset['test'] = SegmentationData(image_paths_file=SEGMENTATION_ROOT + "test.txt")

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
