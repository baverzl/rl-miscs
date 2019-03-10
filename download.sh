#!/bin/sh

echo "Downloading obstacle tower environment"
zip_file=obstacletower_v1.2_linux.zip
wget --verbose https://storage.googleapis.com/obstacle-tower-build/v1.2/$zip_file -O obstacle-tower-challenge/$zip_file
cd obstacle-tower-challenge
unzip $zip_file
