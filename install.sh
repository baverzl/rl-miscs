#!/bin/bash

echo "Get modules - (obstacle-tower-env, obstacle-tower-challenge)"
git submodule init
git submodule update

cd obstacle-tower-env
echo "Installing [obstacle-tower-env]"
pip3 install -e .
cd ../

cd obstacle-tower-challenge
echo "Installing [obstacle-tower-challenge]"
pip3 install -r requirements.txt
cd ../

echo "Downloading obstacle tower environment"
. ./download.sh