#!/bin/bash

echo "Pull submodules"
git submodule init
git submodule update

cd obstacle-tower-env
echo "Install [obstacle-tower-env]"
pip3 install -e .
cd ../

cd obstacle-tower-challenge
echo "Install [obstacle-tower-challenge]"
pip3 install -r requirements.txt
cd ../

. ./download.sh
