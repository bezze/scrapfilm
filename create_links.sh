IDIR=~/scripts/scrapfilm

[[ -d "$IDIR" ]] && rm -r $IDIR
[[ ! -d "$IDIR" ]] && mkdir $IDIR
for file in *.py; do
    ln -s $PWD/$file $IDIR/${file/%.py/} 
done
