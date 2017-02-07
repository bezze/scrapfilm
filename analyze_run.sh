

path=$1

cd $path

for n in {1..3}; do

    cd ${n}_run

    # gunzip ./film_xmol.gz ./vel.dat.gz
    #savescrap 120 ./film_xmol ./vel.dat
    # gzip ./film_xmol ./vel.dat

    gather_hist cm
    gather_kura cm

    cd ..

done
