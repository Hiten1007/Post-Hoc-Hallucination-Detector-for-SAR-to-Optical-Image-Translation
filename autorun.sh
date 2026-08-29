#!/bin/bash
echo "=== STEP 1: DOWNLOADING 4-SEASON DATASET ==="
echo "m1474000" > pass.txt
chmod 600 pass.txt
mkdir -p data/sen12ms/

# Pull all 4 seasons (510 GB) from the official TUM rsync server
rsync -chavzP --password-file=pass.txt rsync://m1474000@dataserv.ub.tum.de/m1474000/ data/sen12ms/

echo "=== STEP 2: STARTING AI TRAINING ==="
# Automatically start training the millisecond the download finishes
python train.py
