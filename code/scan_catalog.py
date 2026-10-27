# -*- coding: utf-8 -*-
"""Scan MaleCNS v1.0 catalog for visual/motion/motor neuron classes
usable as expanded readouts. Output: class counts + which 14 DNs author used.
"""
import pandas as pd

P = r'D:\fly\experiment\data-reference\brain-snapshot-v0\normalized\catalog.feather'
df = pd.read_feather(P)
ct = df['cell_type'].astype(str)

out = []
for pref in ['T4', 'T5', 'LC4', 'LC6', 'LPLC', 'GF', 'MDN', 'MN9',
             'DNp20', 'DNpe017', 'DNa02', 'DNp09', 'sugar', 'lamina',
             'R1', 'R2', 'R6', 'L1', 'L2', 'L3']:
    m = ct[ct.str.startswith(pref, na=False)]
    out.append(f'{pref:10s} types={m.nunique():4d} neurons={len(m):5d}')
print('\n'.join(out))
print()
print('superclass counts:')
print(df['superclass'].fillna('(na)').value_counts().head(20).to_string())
