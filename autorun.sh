#!/bin/bash
echo "=== STEP 1: DOWNLOADING DATASET ==="
echo "m1474000" > pass.txt
chmod 600 pass.txt
mkdir -p data/sen12ms/

BASE="rsync://m1474000@dataserv.ub.tum.de/m1474000"

echo "Downloading Spring season..."
rsync -chavP --password-file=pass.txt $BASE/ROIs1158_spring_lc.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1158_spring_s1.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1158_spring_s2.tar.gz data/sen12ms/

echo "Downloading Summer season..."
rsync -chavP --password-file=pass.txt $BASE/ROIs1868_summer_lc.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1868_summer_s1.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1868_summer_s2.tar.gz data/sen12ms/

echo "Downloading Fall season..."
rsync -chavP --password-file=pass.txt $BASE/ROIs1970_fall_lc.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1970_fall_s1.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs1970_fall_s2.tar.gz data/sen12ms/

echo "Downloading Winter season..."
rsync -chavP --password-file=pass.txt $BASE/ROIs2017_winter_lc.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs2017_winter_s1.tar.gz data/sen12ms/
rsync -chavP --password-file=pass.txt $BASE/ROIs2017_winter_s2.tar.gz data/sen12ms/

echo "=== STEP 1.5: EXTRACTING FILES ==="
find data/sen12ms -name "*.tar.gz" -execdir tar -xzf {} \; -execdir rm {} \;

echo "=== STEP 2: STARTING AI TRAINING ==="
python train.py
