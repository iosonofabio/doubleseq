# vim: fdm=indent
'''
author:     Fabio Zanini
date:       25/08/24
content:    Test data from Luciani, Biro, eLife (https://elifesciences.org/articles/56554#s2) bulk RNA-Seq with a focus on chemokines
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('GTK4agg')
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import scanpy as sc


if __name__ == "__main__":
    
    print('Read bulk RNA-Seq data from paper supplementary')
    fn_bulk = '../data/literature/Tcell_swarming_Luciani_biro/elife-56554-supp1-v3.xlsx'
    df_bulk = pd.read_excel(fn_bulk)
    df_bulk = df_bulk.groupby('Gene_short_name').sum()
    coverage = df_bulk.iloc[:, -6:].sum(axis=0)
    for col in df_bulk.columns[-6:]:
        df_bulk[f'{col}_norm'] = 1.0e6 * df_bulk[col] / coverage[col]
    df_bulk.index = df_bulk.index.str.replace('Fam189b', 'Entrep3')


    print('Read our doublet data')
    fn_atg = '../data/counts/atg_ps_raw.h5ad'
    adata_atg = anndata.read_h5ad(fn_atg)
    adata_atg.var_names = adata_atg.var_names.str.replace('Fam189b', 'Entrep3')

    adata_atg = adata_atg[adata_atg.obs[adata_atg.obs['Comment'] != 'repeatly picking'].index]
    adata_atg.obs['T'] = adata_atg.obs['T'].astype(int)
    adata_atg.obs['DC'] = adata_atg.obs['DC'].astype(int)

    print('Normalise')
    sc.pp.normalize_total(
        adata_atg,
        target_sum=1e6,
        key_added='total',
    )

    # NOTE: there are some cells with 2 T, or 2 T plus a DC, etc. ignore those explicitely
    # There are also 3 negative control wells.

    print('Split')
    adata_T = adata_atg[(adata_atg.obs['T'] == 1) & (adata_atg.obs['DC'] == 0)]
    adata_DC = adata_atg[(adata_atg.obs['T'] == 0) & (adata_atg.obs['DC'] == 1)]
    adata_dou = adata_atg[(adata_atg.obs['T'] == 1) & (adata_atg.obs['DC'] == 1)]
    adatad = {'T': adata_T, 'DC': adata_DC, 'dou':  adata_dou}

    print('Find genes that are only expressed in doublets')
    df_frac = pd.DataFrame([], index=adata_dou.var_names)
    for ct,  adatai in adatad.items():
        df_frac[ct] =  np.asarray((adatai.X > 0).mean(axis=0))[0]

    genes_cands = df_frac.index[df_frac[['T', 'DC']].max(axis=1) == 0]
    genes_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=0))[0], index=genes_cands)
    genes_new = genes_new.loc[genes_new > 0]
    n_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=1))[:, 0], index=adata_dou.obs_names)

    mat_new_genes = df_bulk.loc[genes_new.nlargest(10).index].iloc[:, -6:]
    fig, ax = plt.subplots(figsize=(4, 8))
    sns.heatmap(np.log1p(mat_new_genes), ax=ax)
    fig.tight_layout()
    plt.ion(); plt.show()
    
