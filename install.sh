#!/bin/bash

base=0

for arg
do
  case $arg in
    -base) base=1;;
    *) echo "Usage: `basename $0` [-base]" 1>&2
       exit 1;;
  esac
done

echo "Get modules - (obstacle-tower-env, obstacle-tower-challenge)"
git submodule init
git submodule update

cd submodules/obstacle-tower-env
echo "Installing [obstacle-tower-env]"
#pip install -e .
python setup.py develop
cd ../../

cd submodules/obstacle-tower-challenge
echo "Installing [obstacle-tower-challenge]"
pip install -r requirements.txt
pip install matplotlib
cd ../../

if [ ${base} -eq 1 ];then
  cd submodules/baselines
  echo "Installing [openai/baselines]"
  sudo apt-get update
  sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
  pip3 install tensorflow-gpu==1.8.0
  pip3 install -e .
  cd ../../
fi

echo "Downloading obstacle tower environment"
. ./download.sh
