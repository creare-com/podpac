#!python
import subprocess

with open('zip_package_sizes.txt', 'r') as fid:
    zps = fid.read()

with open('package_sizes.txt', 'r') as fid:
    ps = fid.read()

def parse_ps(ps):
    lns = ps.split('\n')
    pkgs = {}
    for ln in lns:
        try:
            parts = ln.split('\t')
            pkgs[parts[1]] = int(parts[0])
        except:
            pass
    return pkgs

pgz = parse_ps(zps)
pg = parse_ps(ps)

data = {}
for p, s in pgz.items():
    os = pg.get(p[:-4], 0)
    data[p] = {"zip_size": s, "size": os, 'ratio':os*1.0/s}

sdata = sorted(data.items(), key=lambda t: t[1]['ratio'])

zipsize = data['podpac_dist.zip']['zip_size']
totsize = sum([pg[k] for k in pg if (k + '.zip') not in pgz.keys()])
pkgs = []
for val in sdata[::-1]:
    if val[0] == 'podpac_dist.zip':
        continue
    key = val[0]
    pkgs.append(key)

    zipsize += data[key]['zip_size']
    totsize += data[key]['size']

    if (zipsize > 50000 or totsize > 250000):
        k = pkgs.pop()
        zipsize -= data[k]['zip_size']
        totsize -= data[k]['size']

core = [k[:-4] for k in pkgs if k != 'podpac_dist.zip']
deps = [k[:-4] for k in data if k[:-4] not in core and k != 'podpac_dist.zip']
dep_size = sum([data[k+'.zip']['size'] for k in deps])
dep_size_zip = sum([data[k+'.zip']['zip_size'] for k in deps])

# add core to podpac_dist.zip
cmd = ['zip', '-9', '-rq', 'podpac_dist.zip'] + core
subprocess.call(cmd)
cmd = ['zip', '-9', '-rqy', 'podpac_deps.zip'] + deps
subprocess.call(cmd)
