import os, sys
import csv


def cut(line):
    item = line.split()
    dummy=[]
    dummy.append(bmk)
    dummy.append(item[1])
    dummy.extend(item[-3:])
    return dummy


def dump2csv(fname):
    global bmk 
    if "csv" in fname or "parse.py" == fname or ".info" in fname:
        return
    
    knot = ['bmk', 'metric', 'min', 'max', 'average']
    # extarct sm abd bmk from file name 
    knot.append('\n')
    
    bmk = fname[:fname.find('.')]

    # run file 
    with open(fname, 'r') as src:
        for line in src:
            if 'ipc' in line or  'gld_throughput' in line or 'gst_throughput'  in line:
                knot.extend(cut(line))
                knot.append('\n')

            else:
                continue
    with open(bmk+'.csv', 'wb') as dest:
        wr = csv.writer(dest, quoting=csv.QUOTE_ALL)
        wr.writerow(knot)

files = os.listdir('.')
for fname in files:
    dump2csv(fname)

