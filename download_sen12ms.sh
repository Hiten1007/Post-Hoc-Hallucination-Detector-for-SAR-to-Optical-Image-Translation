#!/bin/bash
echo "Creating dataset directory..."
mkdir -p data/sen12ms
cd data/sen12ms

echo "Downloading SEN12MS Dataset from TUM (This will take a while)..."

wget -O ROIs1158_spring.tar.gz "https://dataserv.ub.tum.de/s/m1568244/download?path=%2F&files=ROIs1158_spring.tar.gz"
wget -O ROIs1868_summer.tar.gz "https://dataserv.ub.tum.de/s/m1568244/download?path=%2F&files=ROIs1868_summer.tar.gz"
wget -O ROIs1970_fall.tar.gz "https://dataserv.ub.tum.de/s/m1568244/download?path=%2F&files=ROIs1970_fall.tar.gz"
wget -O ROIs2017_winter.tar.gz "https://dataserv.ub.tum.de/s/m1568244/download?path=%2F&files=ROIs2017_winter.tar.gz"

echo "Extracting archives..."
tar -xzf ROIs1158_spring.tar.gz
tar -xzf ROIs1868_summer.tar.gz
tar -xzf ROIs1970_fall.tar.gz
tar -xzf ROIs2017_winter.tar.gz

echo "Cleaning up compressed tar files to save EBS hard drive space..."
rm *.tar.gz

echo "Download and extraction 100% complete!"
