import torch
import pytorch_lightning as pl

class QMLModel(torch.nn.Module):
    def __init__(self, qnode, weights, config):
        super().__init__()
        self.qnode = qnode
        self.nr_classes = config["num_classes"]
        self.register_parameter("weights", torch.nn.Parameter(weights))

    def forward(self, input):
        probs = self.qnode(self.weights, input)
        
        if len(probs.shape) == 1:
            return probs[:self.nr_classes][None]

        else:
            return probs[:, :self.nr_classes]
        
class TrainingModule(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        X, Y = batch
        predictions = self.model(X)
        loss = self.loss_function(predictions, Y)
        acc = accuracy(threshold(predictions), Y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        X, Y = batch
        predictions = self.model(X)
        loss = self.loss_function(predictions, Y)
        acc = accuracy(threshold(predictions), Y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch):
        X, Y = batch
        predictions = self.model(X)
        loss = self.loss_function(predictions, Y)
        acc = accuracy(threshold(predictions), Y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    # TODO: Factorize code in steps to reduce footprint
    def step(self, batch):
        X, Y = batch
        predictions = self.model(X)
        loss = self.loss_function(predictions, Y)
        acc = accuracy(threshold(predictions), Y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters())

# Choose more likely prediction from probability distribution
def threshold(prediction):
    _, indices = torch.max(prediction, dim=1)
    return indices

# Determine accuracy of predictions
def accuracy(predictions, labels):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc