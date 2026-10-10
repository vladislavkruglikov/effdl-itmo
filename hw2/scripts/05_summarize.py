#!/usr/bin/env python3
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import sys
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/mistral_nemo_h100_5rep.csv')
groups = defaultdict(list)
with path.open() as f:
    for row in csv.DictReader(f):
        groups[(row['method'], row['kind'])].append((float(row['runtime_ms']), float(row['op_mem_mb'])))
print('method,kind,n,mean_ms,sd_ms,mean_mb')
for (method, kind), values in groups.items():
    t = [v[0] for v in values]
    print(f'{method},{kind},{len(t)},{statistics.mean(t):.2f},{statistics.stdev(t) if len(t)>1 else 0:.2f},{statistics.mean(v[1] for v in values):.2f}')
