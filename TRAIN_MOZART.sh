#!/bin/bash
# Ultra-simple Mozart training script
# Just runs the existing training code after setting up the data

set -e

echo "============================================================"
echo "Mozart ChordGNN Finetuning - FINAL SETUP"
echo "============================================================"
echo ""

# Make sure we're in the right directory
cd /Users/kavyabhat/IWold

# Activate environment
source ~/miniconda3/bin/activate chordgnn

echo "Step 1: Setting up Mozart dataset in AugmentedNet format..."

# Create a "mozart" collection in the AugmentedNet cache
MOZART_CACHE="$HOME/.chordgnn/AugmentedNetChordDataset/dataset-mozart"
mkdir -p "$MOZART_CACHE"/{training,validation,test}

# Copy TSVs
cp /Users/kavyabhat/mozart_dataset/training/*.tsv "$MOZART_CACHE/training/"
cp /Users/kavyabhat/mozart_dataset/validation/*.tsv "$MOZART_CACHE/training/" # validation -> training
cp /Users/kavyabhat/mozart_dataset/test/*.tsv "$MOZART_CACHE/test/"

echo "✓ Data copied to: $MOZART_CACHE"
echo "  Training: $(ls $MOZART_CACHE/training/*.tsv | wc -l) files"
echo "  Test: $(ls $MOZART_CACHE/test/*.tsv | wc -l) files"
echo ""

echo "Step 2: Starting training..."
echo "  This will train on YOUR Mozart data with transposition augmentation"
echo "  Press Ctrl+C to stop"
echo ""

# Run the existing training script
# We'll modify the dataset path at runtime
python3 << 'EOF'
import chordgnn as st
import torch, random, os

# Monkey patch the dataset download to use our Mozart folder
original_augmented = st.data.datasets.chord.AugmentedNetChordDataset

class MozartAugmented(original_augmented):
    def __init__(self, raw_dir=None, *args, **kwargs):
        # Force it to use our Mozart dataset folder
        mozart_dir = os.path.expanduser("~/.chordgnn/AugmentedNetChordDataset-Mozart")
        if os.path.exists(os.path.expanduser("~/.chordgnn/AugmentedNetChordDataset/dataset-mozart")):
            # Create a symlink
            if not os.path.exists(mozart_dir):
                os.makedirs(os.path.dirname(mozart_dir), exist_ok=True)
                os.symlink(
                    os.path.expanduser("~/.chordgnn/AugmentedNetChordDataset/dataset-mozart"),
                    os.path.join(mozart_dir, "dataset")
                )
        super().__init__(raw_dir=mozart_dir, *args, **kwargs)

st.data.datasets.chord.AugmentedNetChordDataset = MozartAugmented

# Now run standard training
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.manual_seed(0)
random.seed(0)

print("Loading Mozart dataset...")
datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=4, include_synth=False, num_tasks=11,
    collection="all", batch_size=4, version="v1.0.0")

print(f"✓ Dataset loaded!")
print(f"  Training: {len(datamodule.dataset_train)} samples")
print(f"  Test: {len(datamodule.dataset_test)} samples")
print()

model = st.models.chord.ChordPrediction(
    datamodule.features, 256, datamodule.tasks, 1,
    lr=0.001, dropout=0.44, weight_decay=0.0035,
    use_nade=False, use_jk=False, use_rotograd=False,
    use_gradnorm=False, device=0 if torch.cuda.is_available() else "cpu",
    weight_loss=True)

checkpoint = ModelCheckpoint(save_top_k=1, monitor="global_step", mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=[0] if torch.cuda.is_available() else None,
    callbacks=[checkpoint, early_stop])

print("Training...")
trainer.fit(model, datamodule)
print(f"\n✓ Training complete! Model: {checkpoint.best_model_path}")

trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)
EOF

echo ""
echo "============================================================"
echo "DONE!"
echo "============================================================"
