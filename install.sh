#!/bin/bash

echo "Get modules - (obstacle-tower-env, obstacle-tower-challenge)"
git submodule init
git submodule update

cd submodules/obstacle-tower-env
echo "Installing [obstacle-tower-env]"
pip3 install -e .
cd ../../

cd submodules/obstacle-tower-challenge
echo "Installing [obstacle-tower-challenge]"
pip3 install -r requirements.txt
pip3 install matplotlib
cd ../../

cd submodules/baselines
echo "Installing [openai/baselines]"
sudo apt-get update
sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip3 install tensorflow-gpu
pip3 install -e .
cd ../../


echo "Downloading obstacle tower environment"
. ./download.sh
