import os
import glob

filenames = glob.glob('../geom/*-*.xyz')
for filename in sorted(filenames):
    root = filename[8:-4]
    print(root)
    os.chdir(root)
    os.system('terachem input.dat >& output.dat')
    os.chdir('..')
