#!/bin/bash

sudo apt-get update
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y tzdata
sudo apt-get -q install r-base -y --allow-unauthenticated
sudo apt-get -q install libssl-dev -y
sudo apt-get -q install libgmp3-dev  -y --allow-unauthenticated
sudo apt-get -q install git -y
sudo apt-get -q install build-essential  -y --allow-unauthenticated
sudo apt-get -q install libv8-3.14-dev  -y --allow-unauthenticated
sudo apt-get -q install libcurl4-openssl-dev -y --allow-unauthenticated
Rscript -e 'install.packages(c("V8","sfsmisc","clue","randomForest","lattice","devtools","MASS"),repos="http://cran.us.r-project.org")'
Rscript -e 'install.packages("BiocManager"); BiocManager::install(c("SID", "bnlearn"))'
