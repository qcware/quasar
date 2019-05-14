import os
import glob

filenames = glob.glob('../geom/*-*.xyz')
for filename in sorted(filenames):
    root = filename[8:-4]
    os.system('mkdir %s' % root)
    os.system('cp %s %s/geom.xyz' % (filename, root))
    os.system('cp input.dat %s' % (root))
