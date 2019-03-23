#!/bin/sh

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  zip_file=obstacletower_v1.2_linux.zip
elif [[ "$OSTYPE" == "darwin"* ]]; then
  zip_file=obstacletower_v1.2_osx.zip
fi

wget --verbose https://storage.googleapis.com/obstacle-tower-build/v1.2/$zip_file -O submodules/obstacle-tower-challenge/$zip_file
cd submodules/obstacle-tower-challenge
unzip $zip_file
cd ../../
