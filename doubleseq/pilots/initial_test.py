# vim: fdm=indent
'''
author:     Fabio Zanini
date:       18/08/24
content:    Initial test to check out the double-seq data.
'''
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('gtk4agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text
import anndata
import anndataks
import scanpy as sc

if __name__ == "__main__":

    print('Load data')
    fn_atg = '../data/counts/atg_ps_raw.h5ad'
    adata_atg = anndata.read_h5ad(fn_atg)
    adata_atg = adata_atg[adata_atg.obs[adata_atg.obs['Comment'] != 'repeatly picking'].index]
    adata_atg.obs['T'] = adata_atg.obs['T'].astype(int)
    adata_atg.obs['DC'] = adata_atg.obs['DC'].astype(int)

    print('Normalise')
    sc.pp.normalize_total(
        adata_atg,
        target_sum=1e6,
        key_added='total',
    )

    #fn_dc = '../data/counts/primeDC.h5ad'
    #adata_dc = anndata.read_h5ad(fn_dc)

    # NOTE: there are some cells with 2 T, or 2 T plus a DC, etc. ignore those explicitely
    # There are also 3 negative control wells.

    print('Split')
    adata_T = adata_atg[(adata_atg.obs['T'] == 1) & (adata_atg.obs['DC'] == 0)]
    adata_DC = adata_atg[(adata_atg.obs['T'] == 0) & (adata_atg.obs['DC'] == 1)]
    adata_dou = adata_atg[(adata_atg.obs['T'] == 1) & (adata_atg.obs['DC'] == 1)]

    print('Differential Expression')
    de_vs_T = anndataks.compare(adata_dou, adata_T)
    de_vs_T['frac1'] = np.asarray((adata_dou.X > 0).mean(axis=0))[0]
    de_vs_T['frac2'] = np.asarray((adata_T.X > 0).mean(axis=0))[0]
    de_vs_T['delta_frac'] = de_vs_T['frac1'] - de_vs_T['frac2']
    de_vs_DC = anndataks.compare(adata_dou, adata_DC)
    de_vs_DC['frac1'] = np.asarray((adata_dou.X > 0).mean(axis=0))[0]
    de_vs_DC['frac2'] = np.asarray((adata_DC.X > 0).mean(axis=0))[0]
    de_vs_DC['delta_frac'] = de_vs_DC['frac1'] - de_vs_DC['frac2']

    print('Plot some evidence')
    x, y = de_vs_T['delta_frac'], de_vs_DC['delta_frac']
    r = np.sqrt(x**2 + y**2)
    alpha = r / r.max()
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        x, y,
        color='k', alpha=alpha,
    )
    ax.add_patch(plt.Circle((0, 0), 0.05, ec='red', fc='none'))
    ax.grid(True)
    ax.set_xlabel('$\\Delta fr (vs T)$')
    ax.set_ylabel('$\\Delta fr (vs DC)$')
    xymax = max(x.max(), y.max())
    xymin = min(x.min(), y.min())
    ax.set_xlim(1.1 * xymin, 1.1 * xymax)
    ax.set_ylim(1.1 * xymin, 1.1 * xymax)
    df = pd.Series(r, index=de_vs_DC.index).to_frame(name='r')
    df['x'] = x
    df['y'] = y
    texts = []
    for gene, row in df.nlargest(50, 'r').iterrows():
        xi = row['x']
        yi = row['y']
        text = ax.text(xi, yi, gene, ha='center', va='bottom')
        texts.append(text)
    adjust_text(texts)
    fig.tight_layout()
    plt.ion(); plt.show()
