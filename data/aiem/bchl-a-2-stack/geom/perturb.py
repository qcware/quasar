import re

def read_xyz(filename):

    lines = open(filename).readlines()[2:]
    geom = []
    for line in lines:
        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
        geom.append([
            mobj.group(1),
            float(mobj.group(2)),
            float(mobj.group(3)),
            float(mobj.group(4)),
            ])
    return geom

def write_xyz(filename, geom):

    fh = open(filename, 'w')
    fh.write('%d\n\n' % len(geom))
    for atom in geom:
        fh.write('%-2s %10.5f %10.5f %10.5f\n' % tuple(atom))

if __name__ == '__main__':

    import sys
    root = sys.argv[1]
    index = int(sys.argv[2])
    dim = sys.argv[3]
    delta = float(sys.argv[4])

    if dim == 'x':
        d = 0
    elif dim == 'y':
        d = 1
    elif dim == 'z':
        d = 2
    else:
        raise RuntimeError('Invalid dim: %s' % dim)

    geom = read_xyz('%s.xyz' % (root))
    geom[index][d+1] += delta
    write_xyz('%s-%d-%s-%s%s.xyz' % (root, index, dim, 'p' if delta > 0.0 else 'm', abs(delta)), geom)
