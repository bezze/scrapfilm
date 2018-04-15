#!/usr/bin/env python3
# Use PIL to save some image metadata

def add_meta(fid, METADATA):
    # fid is string with path
    # METADATA is a dictionary with format field : value
    import os

    fid = os.path.realpath(fid)
    ext = fid.split('.')[-1]

    if ext == "png":
        """ PIL viene en el paquete Pillow """
        from PIL import Image
        from PIL import PngImagePlugin
        im = Image.open(fid)
        meta = PngImagePlugin.PngInfo()

        for x in METADATA:
            meta.add_text(x, METADATA[x])
        im.save(fid, "png", pnginfo=meta)

    elif ext == "svg":
        with  open(fid,'a') as im:
            im.write("\n<!-- \n")
            for x in METADATA:
                im.write( x+' : '+METADATA[x]+'\n' )
            im.write("-->")
    else:
        print("Extension not recognized!")

def read_meta(fid):
    # fid is a file descriptor or a string
    # METADATA is a dictionary with format field : value

    """ PIL viene en el paquete Pillow """
    from PIL import Image
    from PIL import PngImagePlugin

    im = Image.open(fid)
    for field in im.info:
        print(field,' : ',im.info[field])

def produce_meta():

    import os, sys, time


    script_path = os.path.abspath(sys.argv[0])
    mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime = os.stat(script_path)

    METADATA = {}
    METADATA['script_path'] = script_path
    METADATA['script_size'] = str(size)
    METADATA['last_edit'] = time.ctime(mtime)
    METADATA['last_cmd'] = " ".join(sys.argv)

    return METADATA

if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        read_meta(f)
