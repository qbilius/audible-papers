from pathlib import Path
import json, csv
import shutil

import numpy as np
import torch, torch.utils.data
from torch import nn
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter
import tiktoken

import dataset, model


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 block_size: int,
                 batch_size: int,
                 num_workers: int
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.enc = tiktoken.get_encoding('gpt2')

    def _dataloader(self, stage: str):
        return torch.utils.data.DataLoader(
            dataset.Dataset(
                self.data_dir,
                block_size=self.hparams.block_size,
                stage=stage,
            ),
            batch_size=self.hparams.batch_size,
            shuffle=stage == 'train',
            num_workers=self.hparams.num_workers,
            pin_memory=False
        )

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def predict_dataloader(self):
        samples = ...  # TODO
        return self._dataloader(samples, 'predict')


class Model(pl.LightningModule):

    def __init__(self,
                 block_size: int,
                 n_layer: int,
                 n_head: int,
                 n_embed: int,
                 dropout: float,
                 bias: bool,
                 beta1: float,
                 beta2: float,
                 learning_rate=1e-3,
                 fc_lr_ratio=10,
                 use_adamw=True,
                 momentum=.9,
                 weight_decay=1e-4,
                 use_scheduler=True,
                 eta_min=1e-4,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model.GPT(
            block_size=block_size,
            vocab_size=50304,
            n_layer=n_layer,
            n_head=n_head,
            n_embed=n_embed,
            dropout=dropout,
            bias=bias
        )

        self.probs = nn.Softmax(dim=1)
        self.loss = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

        self.enc = tiktoken.get_encoding('gpt2')

    def forward(self, x):
        out = self.model(x)
        return out

    def _evaluate(self, batch, stage):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss(logits, y)
        self.log(f'{stage}_loss', loss, prog_bar=True)

        acc = self.train_acc if stage == 'train' else self.val_acc
        acc(logits, y)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._evaluate(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, 'val')

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        feats = self.features.features.view(self.features.features.shape[0], -1)
        return feats, self.probs(y_hat)

    def configure_optimizers(self) -> ([torch.optim.AdamW], [torch.optim.lr_scheduler.CosineAnnealingLR]):
        # start with all of the candidate parameters
        param_dict: dict[str, nn.Parameter] = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params: list[nn.Parameter] = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params: list[nn.Parameter] = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups: [dict[str, list[nn.Parameter] | float]] = [
            {'params': decay_params,
             'weight_decay': self.hparams.weight_decay},
            {'params': nodecay_params,
             'weight_decay': 0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            fused=self.device == 'cuda'
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.hparams.eta_min)
        return [optimizer], [scheduler]


class PredictionWriter(BasePredictionWriter):

    def __init__(self):
        super().__init__('epoch')
        self.targets = None
        self.header = ['id', 'phase', 'split', 'phenotype', 'diversity', 'train_phenotype']

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        assert len(batch_indices) == 1  # there should only be a single epoch
        batch_indices = batch_indices[0]

        dataset = trainer.predict_dataloaders.dataset
        # Need to load val to get only the phenotypes used during training
        if self.targets is None:
            samples = json.load(open(Path(dataset.data_dir) / 'splits' / f'{trainer.datamodule.hparams["data_prefix"]}_val.json'))
            self.targets = np.unique([s['train_phenotype'] for s in samples if s['split'] == 'val']).tolist()

        f = open(Path(trainer.ckpt_path).parents[1] / 'predictions.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(self.header + ['well'] + list(self.targets))
        features = []
        for batch, prediction_batch in zip(batch_indices, predictions):
            # if there is a single sample, features were flattened
            if prediction_batch[0].ndim == 1:
                prediction_batch[0] = prediction_batch[0].unsqueeze(0)
            for bi, feats, prediction in zip(batch, *prediction_batch):
                sample = dataset.samples[bi]
                well = '-'.join(sample['id'].split('-')[-1].split('_')[:-1])
                if sample['split'] == 'predict':  # doesn't have train_phenotype
                    sample['train_phenotype'] = ''
                header_data = [sample[c] for c in self.header]
                writer.writerow(header_data + [well] + prediction.detach().numpy().tolist())
                features.append(feats.cpu().numpy())
        f.close()
        np.save(Path(f.name).parent / 'features.npy', features)


class Trainer(pl.Trainer):
    @property
    def log_dir(self) -> str | None:
        """Quick fix for config.yaml to be saved in log_dir rather than save_dir"""
        if len(self.loggers) > 0:
            dirpath: str | None = self.loggers[0].log_dir
        elif self.ckpt_path is not None:
            dirpath = str(Path(self.ckpt_path).parents[1])
        else:
            dirpath = self.default_root_dir
        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--no_output", action='store_true', default=False)

        env_file = Path('.env')
        if env_file.exists():
            with env_file.open() as f:
                root = f.read().strip()
            parser.set_defaults({'trainer.default_root_dir': root})

        parser.link_arguments('trainer.default_root_dir', 'trainer.logger.init_args.save_dir')
        parser.link_arguments('trainer.default_root_dir', 'data.data_dir')
        parser.link_arguments('model.block_size', 'data.block_size')

    def parse_arguments(self, parser, args):
        super().parse_arguments(parser, args)
        if self.config['subcommand'] == 'predict':
            # Do not save config.yaml when predicting
            self.save_config_callback = None

    def after_fit(self):
        if self.config['fit']['no_output']:
            shutil.rmtree(self.trainer.log_dir)


if __name__ == '__main__':
    MyLightningCLI(
        Model,
        DataModule,
        trainer_class=Trainer
    )
