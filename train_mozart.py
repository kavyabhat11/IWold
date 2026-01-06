#!/usr/bin/env python3
"""
Train ChordGNN on Mozart piano sonata data only.
Usage: python train_mozart.py --gpus 0 --n_epochs 50
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

# Import the base dataset class
from chordgnn.data.dataset import BuiltinDataset
from chordgnn.data.datasets.chord import ChordGraphDataset, data_to_graph


class MozartDataset(BuiltinDataset):
    """Mozart Piano Sonata Dataset - local TSV files."""

    def __init__(self, raw_dir="./mozart_dataset", force_reload=False, verbose=True):
        self.local_path = raw_dir
        super(MozartDataset, self).__init__(
            name="MozartDataset",
            url=None,  # No download needed
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        # No download - using local data
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(f"Mozart dataset not found at {self.local_path}")

    def process(self, subset=""):
        """Find all TSV files in the Mozart dataset."""
        self.scores = []
        for root, dirs, files in os.walk(self.raw_path):
            if subset and not root.endswith(subset):
                continue
            for file in files:
                if file.endswith(".tsv"):
                    self.scores.append(os.path.join(root, file))

    def has_cache(self):
        return os.path.exists(self.raw_path)


class MozartChordGraphDataset(ChordGraphDataset):
    """Mozart Chord Graph Dataset for ChordGNN."""

    def __init__(self, raw_dir="./mozart_dataset", force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11,
                 collection="all", max_size=512):
        dataset_base = MozartDataset(raw_dir=raw_dir)
        self.collection = collection
        self.include_synth = include_synth
        self.prob_pieces = []

        # Task configuration (same as AugmentedNet)
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22,
                    "quality": 16, "inversion": 4, "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22,
                    "quality": 16, "inversion": 4, "root": 35, "romanNumeral": 76,
                    "hrhythm": 2, "pcset": 94, "bass": 35}
        else:
            from chordgnn.utils.chord_representations import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}

        super(MozartChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="MozartChordGraphDataset",
            raw_dir=os.path.join(raw_dir, ".processed"),
            force_reload=force_reload,
            verbose=verbose)

    def _process_score(self, score_fn):
        """Process a Mozart TSV file into graph format."""
        name = os.path.splitext(os.path.basename(score_fn))[0]

        # Determine collection (training/validation/test) from directory
        if "training" in score_fn:
            collection = "training"
        elif "validation" in score_fn:
            collection = "training"  # Treat validation as training
        elif "test" in score_fn:
            collection = "test"
        else:
            collection = "training"

        # Process the TSV file
        from chordgnn.utils.chord_representations import time_divided_tsv_to_part

        if collection == "test":
            # Test: no transposition
            note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False)
            data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
        else:
            # Training: with transposition augmentation (automatic!)
            x = time_divided_tsv_to_part(score_fn, transpose=True)
            for i, (note_array, labels) in enumerate(x):
                data_to_graph(note_array, labels, collection,
                            (name + "-{}".format(i) if i > 0 else name),
                            save_path=self.save_path)
        return


# ============================================================================
# Training script
# ============================================================================

parser = argparse.ArgumentParser(description="Train ChordGNN on Mozart data")
parser.add_argument('--gpus', type=str, default="0", help="GPU IDs (e.g., '0' or '0,1')")
parser.add_argument('--n_layers', type=int, default=1, help="Number of GNN layers")
parser.add_argument('--n_hidden', type=int, default=256, help="Hidden units")
parser.add_argument('--dropout', type=float, default=0.44, help="Dropout rate")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0035, help="Weight decay")
parser.add_argument('--num_workers', type=int, default=4, help="Data loader workers")
parser.add_argument('--n_epochs', type=int, default=50, help="Number of epochs")
parser.add_argument('--mozart_data', type=str, default="./mozart_dataset", help="Mozart dataset path")
parser.add_argument('--use_wandb', action="store_true", help="Use Weights & Biases logging")
parser.add_argument('--force_reload', action="store_true", help="Force reload dataset")

# Reproducibility
torch.manual_seed(0)
random.seed(0)

args = parser.parse_args()

# Parse GPU configuration
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

name = f"Mozart-{args.n_layers}x{args.n_hidden}-lr={args.lr}-wd={args.weight_decay}-dr={args.dropout}"

# Logger
if args.use_wandb:
    wandb_logger = WandbLogger(
        log_model=True,
        project="mozart_finetune",
        name=name)
    logger = wandb_logger
else:
    logger = True  # Use default logger

print(f"\n{'='*70}")
print(f"ChordGNN Finetuning on Mozart Piano Sonatas")
print(f"{'='*70}")
print(f"Dataset: {args.mozart_data}")
print(f"Epochs: {args.n_epochs}")
print(f"Model: {args.n_layers} layers × {args.n_hidden} hidden units")
print(f"Learning rate: {args.lr}")
print(f"Batch size: {args.batch_size}")
print(f"Device: {'GPU ' + args.gpus if devices else 'CPU'}")
print(f"{'='*70}\n")

# Create Mozart dataset
print("Loading Mozart dataset...")

# Create processed data directory
processed_dir = os.path.join(args.mozart_data, ".processed", "MozartChordGraphDataset")
os.makedirs(processed_dir, exist_ok=True)

mozart_dataset = MozartChordGraphDataset(
    raw_dir=args.mozart_data,
    force_reload=args.force_reload,
    nprocs=args.num_workers,
    num_tasks=11,
    collection="all"
)

# Create datamodule (manually construct to use Mozart dataset)
datamodule = st.data.AugmentedGraphDatamodule.__new__(st.data.AugmentedGraphDatamodule)
datamodule.__dict__.update({
    'bucket_boundaries': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'force_reload': args.force_reload,
    'normalize_features': True,
    'version': 'v1.0.0'
})

datamodule.datasets = [mozart_dataset]
datamodule.tasks = mozart_dataset.tasks
datamodule.features = mozart_dataset.features

# Setup datamodule (creates train/val/test splits)
datamodule.setup()

print(f"✓ Dataset loaded:")
print(f"  Training samples: {len(datamodule.dataset_train)} (with transposition augmentation)")
print(f"  Validation/Test samples: {len(datamodule.dataset_test)}")
print(f"  Features: {datamodule.features}")
print(f"  Tasks: {list(datamodule.tasks.keys())}\n")

# Create model
print("Creating model...")
model = st.models.chord.ChordPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers,
    lr=args.lr, dropout=args.dropout, weight_decay=args.weight_decay,
    use_nade=False, use_jk=False, use_rotograd=False, use_gradnorm=False,
    device=dev, weight_loss=True)

print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")

# Callbacks
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="global_step",
    mode="max",
    filename="mozart-{epoch:02d}-{global_step}"
)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.02,
    patience=10,
    verbose=True,
    mode="min"
)

# Trainer
trainer = Trainer(
    max_epochs=args.n_epochs,
    accelerator="auto",
    devices=devices,
    num_sanity_val_steps=1,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    reload_dataloaders_every_n_epochs=5,
)

# Train
print(f"{'='*70}")
print("Starting training...")
print(f"{'='*70}\n")

trainer.fit(model, datamodule)

# Test
print(f"\n{'='*70}")
print("Testing on held-out Mozart pieces...")
print(f"{'='*70}\n")

trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)

print(f"\n{'='*70}")
print("✓ Training Complete!")
print(f"{'='*70}")
print(f"Best model saved at: {checkpoint_callback.best_model_path}")
print(f"{'='*70}\n")
