#!/bin/bash
# Complete pipeline to finetune ChordGNN on Mozart data

set -e  # Exit on error

echo "=========================================="
echo "Mozart ChordGNN Finetuning Pipeline"
echo "=========================================="
echo ""

# Step 1: Convert XML+TXT to TSV
echo "Step 1: Converting Mozart XML+TXT files to TSV format..."
python convert_mozart_to_tsv.py \
    --data_dir ./IWold/data/mozart \
    --output_dir ./mozart_tsv

echo ""

# Step 2: Split into train/val/test
echo "Step 2: Splitting dataset (30 train / 5 val / rest test)..."
python split_mozart_data.py \
    --tsv_dir ./mozart_tsv \
    --output_dir ./mozart_dataset \
    --n_train 30 \
    --n_val 5

echo ""

# Step 3: Create custom dataset class (copying to chordgnn/data/datasets/)
echo "Step 3: Setting up Mozart dataset loader..."
cat > ./IWold/chordgnn/data/datasets/mozart_dataset.py << 'EOF'
import os
from chordgnn.data.dataset import BuiltinDataset

class MozartDataset(BuiltinDataset):
    """Mozart Piano Sonata Dataset."""

    def __init__(self, raw_dir="./mozart_dataset", force_reload=False, verbose=True):
        self.mozart_path = raw_dir
        super(MozartDataset, self).__init__(
            name="MozartDataset",
            url=None,  # Local dataset
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        # No download needed - local dataset
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
EOF

echo "  ✓ Mozart dataset class created"
echo ""

# Step 4: Launch training
echo "Step 4: Starting finetuning..."
echo "  This will train for 100 epochs on Mozart data only"
echo ""

cd ./IWold

python -m chordgnn.train.chord_prediction \
    --gpus 0 \
    --n_layers 1 \
    --n_hidden 256 \
    --dropout 0.44 \
    --batch_size 4 \
    --lr 0.001 \
    --weight_decay 0.0035 \
    --num_workers 4 \
    --collection all \
    --n_epochs 100 \
    --num_tasks 11 \
    --data_version v1.0.0

echo ""
echo "=========================================="
echo "Finetuning complete!"
echo "=========================================="
