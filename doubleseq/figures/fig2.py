# vim: fdm=indent
'''
author:     Fabio Zanini
date:       18/08/24
content:    Initial test to check out the double-seq data.
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('gtk4agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import anndata
import anndataks
import scanpy as sc
import gseapy as gp

if __name__ == "__main__":

    print('Load data')
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

    print('Compute differential expression between T and DC to get markers')
    df_avg = pd.DataFrame([], index=adata_dou.var_names)
    for ct,  adatai in adatad.items():
        df_avg[ct] =  np.asarray((adatai.X).mean(axis=0))[0]

    pc = 0
    df_avg_plot = np.log(df_avg[['DC', 'T']] + 10**pc)
    df_avg_plot_nz = df_avg_plot.loc[(df_avg[['DC', 'T']] > pc).any(axis=1)]
    annots = ['Cd2', 'Cd6', 'Cd3d', 'Cd247', 'Cd209a', 'Cd83', 'H2-Ab1', 'H2-DMa', 'Apoe', 'S100a9']
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.kdeplot(
        data=df_avg_plot, x='DC', y='T', ax=ax,
        levels=[0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        fill=False,
        zorder=5,
        cmap='viridis',
        clip=(0, 10),
    )
    ax.scatter(
        df_avg_plot_nz['DC'], df_avg_plot_nz['T'], color='steelblue', alpha=0.01,
        zorder=4,
    )
    ax.scatter(
        df_avg_plot.loc[annots, 'DC'], df_avg_plot.loc[annots, 'T'], color='k', alpha=0.9,
        zorder=4,
    )
    texts = [ax.text(df_avg_plot.at[ge, 'DC'], df_avg_plot.at[ge, 'T'], ge) for ge in annots]
    ax.grid(True)
    ax.set_xlabel('Mean expression in DC [log1p(cpm)]')
    ax.set_ylabel('Mean expression in T cells [log1p(cpm)]')
    xx = np.linspace(0, 10, 100)
    ax.plot(xx, xx, color='grey')
    ax.plot(xx, xx - np.log(10), color='grey', ls='--')
    ax.plot(xx, xx + np.log(10), color='grey', ls='--')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.tight_layout()
    adjust_text(texts, arrowprops=dict(arrowstyle='->'), max_move=40)
    fig.savefig('../figures/fig2_exp_DC_vs_T.svg')
    fig.savefig('../figures/fig2_exp_DC_vs_T.png', dpi=600)
    plt.ion(); plt.show()

    print('Plot genes that are only expressed in doublets')
    df_frac = pd.DataFrame([], index=adata_dou.var_names)
    for ct,  adatai in adatad.items():
        df_frac[ct] =  np.asarray((adatai.X > 0).mean(axis=0))[0]

    genes_cands = df_frac.index[df_frac[['T', 'DC']].max(axis=1) == 0]
    genes_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=0))[0], index=genes_cands)
    genes_new = genes_new.loc[genes_new > 0]
    n_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=1))[:, 0], index=adata_dou.obs_names)

    print('Some statistics to see how unusual this is')
    ngenes = 30
    stats = {}
    for gene in genes_new.nlargest(ngenes).index:
        counts_pos = int((np.asarray(adata_dou[:, gene].X.todense())[:,0] * adata_dou.obs['total'].values / 1e6).sum())
        counts_neg = 0  # by definition
        cells_pos = adata_dou.n_obs
        cells_neg = adata_T.n_obs + adata_DC.n_obs
        # This is an extreme scenario, in which all molecules ended up in the same category. How likely is this (there is nothing more extreme)?
        log10neg_pval = -counts_pos * np.log10(cells_pos / (cells_pos + cells_neg))
        # Bonferroni correction
        log10neg_pval+= np.log10(adata_dou.n_vars)
        stats[gene] = {'counts_pos': counts_pos, 'neglog10_pval_adj': log10neg_pval, 'ndou_exp': genes_new[gene]}
    stats = pd.DataFrame(stats).T
    stats['counts_pos'] = stats['counts_pos'].astype(int)
    stats['ndou_exp'] = stats['ndou_exp'].astype(int)
    stats = stats.sort_values('neglog10_pval_adj', ascending=False)

    print('Plot these genes')
    fig, axs = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
    ax = axs[0]
    x = stats['ndou_exp']
    ax.barh(np.arange(len(x)), x, color='k', zorder=5, alpha=0.7)
    ax.set_xlabel('Expressed in # T/DC doublets')
    ax.set_ylabel('')
    ax.set_yticks(np.arange(len(x)))
    ax.set_yticklabels(x.index)
    ax.grid(True)
    ax.set_ylim(-0.5, len(x) - 0.5)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'])

    ax = axs[1]
    x = stats['neglog10_pval_adj']
    ax.barh(np.arange(len(x)), x, color='tomato', zorder=5, alpha=0.7)
    ax.set_xlabel('$-\log_{10} P_{val}$')
    ax.set_ylabel('')
    ax.set_yticks(np.arange(len(x)))
    ax.set_yticklabels(x.index)
    ax.grid(True)
    ax.set_ylim(-0.5, len(x) - 0.5)

    fig.tight_layout()
    fig.savefig('../figures/fig2_new_genes_stats.svg')
    fig.savefig('../figures/fig2_new_genes_stats.png', dpi=600)
    plt.ion(); plt.show()

    sys.exit()

    print('Pathway analysis on these genes')
    enr_new = gp.enrichr(gene_list=list(genes_new.nlargest(100).index),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )

    from gseapy import Biomart
    bm = Biomart()
    # note the dataset and attribute names are different
    m2h = bm.query(dataset='mmusculus_gene_ensembl',
                   attributes=['ensembl_gene_id','external_gene_name',
                               'hsapiens_homolog_ensembl_gene',
                               'hsapiens_homolog_associated_gene_name'])
    a = genes_new.nlargest(100).index
    a = a[a.isin(m2h['external_gene_name'])]
    genes_new_human = m2h.set_index('external_gene_name').loc[a].dropna()['hsapiens_homolog_associated_gene_name'].values
    enr_new = gp.enrichr(gene_list=list(genes_new_human),
                 gene_sets=['GO_Biological_Process_2023'],
                 organism='mouse',
                 outdir=None,
                )

    print('Clustermap of the doublets for the genes that matter')
    from scipy.spatial.distance import pdist    
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.stats import zscore
    gs = list(genes_new.sort_values(ascending=False).index[:30]) + []
    dhdf = pd.DataFrame(np.asarray(adata_dou[:, gs].X.todense()), index=adata_dou.obs_names, columns=gs)

    dhdf_nz = dhdf.loc[dhdf.sum(axis=1) >  0]
    zscore_cells = zscore(np.log1p(dhdf_nz.values), axis=0)
    pdis = pdist(zscore_cells)
    Z = linkage(pdis, method='average', optimal_ordering=True)

    zscore_genes = zscore(np.log1p(dhdf_nz.values).T, axis=0)
    pdisT = pdist(zscore_genes)
    ZT = linkage(pdisT, method='average', optimal_ordering=True)

    g = sns.clustermap(
        np.log1p(dhdf_nz),
        row_linkage=Z,
        col_linkage=ZT,
        yticklabels=True,
    )
    g.fig.savefig('../figures/fig2_new_genes_clustermap.svg')
    g.fig.savefig('../figures/fig2_new_genes_clustermap.png', dpi=600)

    #print('Plot presence/absence matrix, perhaps clearer')
    #g = sns.clustermap(
    #    (dhdf_nz > 0),
    #    yticklabels=True,
    #    xticklabels=True,
    #)
