#!/usr/bin/bash

path=$1

cd $path

for n in {1..3}; do

    cd ${n}_run

    gunzip ./film_xmol.gz ./vel.dat.gz
    savescrap ./film_xmol ./vel.dat
    gzip ./film_xmol ./vel.dat

    cd ..

done
