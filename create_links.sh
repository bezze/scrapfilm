for file in *.py; do
    rm ~/scripts/${file/%.py/}
    #ln -s ~/py_scripts/scrapfilm/$file ~/scripts/${file/%.py/} 
    ln -s ~/pytest/scrapfilm/$file ~/scripts/${file/%.py/} 
done

