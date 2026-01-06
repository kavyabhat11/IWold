#!/usr/bin/env python3
"""
Simple Mozart finetuning - uses existing ChordGNN code with Mozart TSVs.
Just point the dataset to our Mozart folder in ~/.chordgnn/
"""

import chordgnn as st
import torch
import random
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse

# Override the dataset URL/path to use our Mozart data
from chordgnn.data.dataset import BuiltinDataset

class MozartOnlyDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        # Point to our Mozart dataset in .chordgnn cache
        mozart_path = os.path.expanduser("~/.chordgnn/MozartDataset")
        super(MozartOnlyDataset, self).__init__(
            name="MozartDataset",
            url=None,
            raw_dir=mozart_path,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        # No download - data already there
        pass

    def process(self, subset=""):
        self.scores = []
        for root, dirs, files in os.walk(self.raw_path):
            if root.endswith(subset):
                for file in files:
                    if file.endswith(".tsv"):
                        self.scores.append(os.path.join(root, file))

    def has_cache(self):
        return os.path.exists(self.raw_path)

# Monkey patch to use Mozart dataset instead
original_init = st.data.datasets.chord.AugmentedNetChordDataset.__init__

def mozart_init(self, raw_dir=None, force_reload=False, verbose=True):
    self.dataset_base = MozartOnlyDataset(raw_dir=raw_dir)
    self.dataset_base.process()
    self.max_size = 512
    if verbose:
        print(f"Using Mozart dataset with {len(self.dataset_base.scores)} TSV files")
    from chordgnn.data.datasets.chord import ChordGraphDataset
    ChordGraphDataset.__init__(
        self,
        dataset_base=self.dataset_base,
        max_size=self.max_size,
        nprocs=4,
        name="MozartDataset",
        raw_dir=os.path.expanduser("~/.chordgnn/MozartChordGraphDataset"),
        force_reload=force_reload,
        verbose=verbose)

st.data.datasets.chord.AugmentedNetChordDataset.__init__ = mozart_init

# Now use the regular training code
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.44)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0035)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--use_wandb', action="store_true")

torch.manual_seed(0)
random.seed(0)

args = parser.parse_args()

if isinstance(eval(args.gpus), int):
    if eval(args.gpus) >= 0:
        devices = [eval(args.gpus)]
        dev = devices[0]
    else:
        devices = None
        dev = "cpu"
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    dev = None

name = f"Mozart-{args.n_layers}x{args.n_hidden}-lr={args.lr}"

if args.use_wandb:
    logger = WandbLogger(log_model=True, project="mozart_finetune", name=name)
else:
    logger = True

print(f"\n{'='*70}")
print(f"Finetuning ChordGNN on Mozart Piano Sonatas")
print(f"{'='*70}\n")

# Use standard datamodule but it will load Mozart data
datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=args.num_workers,
    include_synth=False,
    num_tasks=11,
    collection="all",
    batch_size=args.batch_size,
    version="v1.0.0")

model = st.models.chord.ChordPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers,
    lr=args.lr, dropout=args.dropout, weight_decay=args.weight_decay,
    use_nade=False, use_jk=False, use_rotograd=False, use_gradnorm=False,
    device=dev, weight_loss=True)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="global_step", mode="max")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=10, mode="min")

trainer = Trainer(
    max_epochs=args.n_epochs,
    accelerator="auto",
    devices=devices,
    num_sanity_val_steps=1,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    reload_dataloaders_every_n_epochs=5)

print(f"Training for {args.n_epochs} epochs...")
print(f"Training samples: {len(datamodule.dataset_train)}")
print(f"Test samples: {len(datamodule.dataset_test)}\n")

trainer.fit(model, datamodule)

print(f"\nTesting...")
trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)

print(f"\n✓ Complete! Best model: {checkpoint_callback.best_model_path}\n")
