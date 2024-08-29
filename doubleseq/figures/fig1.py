# vim: fdm=indent
'''
author:     Fabio Zanini
date:       18/08/24
content:    Initial test to check out the time-seq data.
'''
import os
import sys
import argparse
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


def load_and_fix_data():
    fn = '../data/counts/primeDC.h5ad'
    adata = anndata.read_h5ad(fn)

    genes = adata.var_names.value_counts().index
    genes = genes.sort_values()
    if genes[0] == '':
        genes = genes[1:]
    genes_ser = pd.Series(np.arange(len(genes)), index=genes)
    X0 = adata.X.toarray()
    X = np.zeros((adata.n_obs, len(genes)), np.float32)
    for gene, vec in zip(adata.var_names, X0.T):
        if gene == '':
            continue
        X[:, genes_ser.at[gene]] += vec
    X = type(adata.X)(X)

    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame([], index=pd.Index(genes, name='Gene')),
        obs=adata.obs,
    )
    return adata


if __name__ == "__main__":

    pa = argparse.ArgumentParser()
    pa.add_argument('--plot', action='store_true')
    args = pa.parse_args()

    print('Load data')
    adata = load_and_fix_data()
    adata.var_names = adata.var_names.str.replace('Fam189b', 'Entrep3')

    print('Normalise')
    sc.pp.normalize_total(
        adata,
        target_sum=1e6,
        key_added='total',
    )

    if args.plot:
        print('Plot distributions of incubation times')
        fig, ax = plt.subplots(figsize=(3, 3))
        x = adata.obs['IncubationTime'].values
        ax.ecdf(x, complementary=True, color='k')
        ax.set_xlabel('Time')
        ax.grid(True)
        fig.tight_layout()
        plt.ion(); plt.show()

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        tmp = adata.obs['IncubationTime'].value_counts().sort_index()
        y = tmp.values
        x = np.arange(len(y))
        ticks = tmp.index
        ax.bar(x, y, color='k', alpha=0.85, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlabel('Time [mins]')
        ax.set_ylabel('# cells')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('../figures/figure2/sampling_time.svg')
        fig.savefig('../figures/figure2/sampling_time.png', dpi=600)
        plt.ion(); plt.show()

    print('Correlate time and gene expression')
    from scipy.stats import pearsonr, spearmanr
    rs = []
    ts = adata.obs['IncubationTime'].values
    for i, gene in enumerate(adata.var_names):
        if (i % 1000)  == 0:
            print(i, end='\r')
        vec = adata.X[:, i].toarray()[:, 0]
        if vec.max() == 0:
            r = np.nan
        else:
            r = pearsonr(ts, vec)[0]
        rs.append(r)
    print('        ')
    adata.var['r_time'] = rs
    adata.var['r_time_zeroed'] = rs.copy()
    adata.var.loc[np.isnan(adata.var['r_time']), 'r_time_zeroed'] = 0

    if args.plot:
        print('Plot correlations')
        fig, ax = plt.subplots(figsize=(4.5, 3.7))
        x = adata.var['r_time_zeroed'].values
        ax.ecdf(x, complementary=True, color='k', lw=2)
        ax.set_ylim(1e-4, 1-1e-4)
        ax.set_yscale('logit')
        ax.grid(True)
        ax.set_xlabel('Pearson r expression vs time')
        ax.set_ylabel('Fraction of genes with correlation > x')
        ax.text(
            0.95, 0.95,
            '\n'.join(list(adata.var['r_time_zeroed'].nlargest(10).index)),
            ha='right', va='top', transform=ax.transAxes,
            bbox=dict(ec='tomato', fc='#FCC', pad=5),
        )
        ax.text(
            0.05, 0.05,
            '\n'.join(list(adata.var['r_time_zeroed'].nsmallest(10).index)),
            ha='left', va='bottom', transform=ax.transAxes,
            bbox=dict(ec='navy', fc='#CCF', pad=5),
        )
        fig.tight_layout()
        fig.savefig('../figures/figure2/cumulative_correlation_time.svg')
        fig.savefig('../figures/figure2/cumulative_correlation_time.png', dpi=600)
        plt.ion(); plt.show()

    print('Pathway analysis on these genes')
    enr_new = gp.enrichr(gene_list=list(adata.var['r_time_zeroed'].nlargest(100).index),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    print(enr_new.results.iloc[:10])

    from gseapy import Biomart
    bm = Biomart()
    # note the dataset and attribute names are different
    m2h = bm.query(dataset='mmusculus_gene_ensembl',
                   attributes=['ensembl_gene_id','external_gene_name',
                               'hsapiens_homolog_ensembl_gene',
                               'hsapiens_homolog_associated_gene_name'])
    a = adata.var['r_time_zeroed'].nlargest(200).index
    a = a[a.isin(m2h['external_gene_name'])]
    genes_new_human = m2h.set_index('external_gene_name').loc[a].dropna()['hsapiens_homolog_associated_gene_name'].values
    enr_new_go = gp.enrichr(gene_list=list(genes_new_human),
                 gene_sets=['GO_Biological_Process_2023'],
                 organism='mouse',
                 outdir=None,
                )

    print('Check expression of individual genes over time in bins')
    # The cells are labelled as times: 0 (24 cells), 10/20/30 (12 cells each)
    def plot_timecourse_gene(gene, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        ts_unique = adata.obs['IncubationTime'].value_counts().sort_index().index
        colors = plt.get_cmap('copper')(np.linspace(0, 1, len(ts_unique)))
        cdict = dict(zip(ts_unique, colors))
        data = adata.obs['IncubationTime'].to_frame()
        data['Expression [cpm]'] = adata[:, gene].X.toarray()[:, 0] + 1
        sns.swarmplot(
            data=data, x='IncubationTime', y='Expression [cpm]',
            hue='IncubationTime', palette=cdict,
            s=5,
            legend=False,
            edgecolor='k',
            linewidth=1.1,
            ax=ax)
        ax.set_yscale('log')
        ax.set_title(gene)
        #ax.text(0.05, 0.95, gene,
        #        ha='left', va='top', transform=ax.transAxes,
        #        bbox=dict(ec='grey', fc='white', pad=5))
        ax.grid(True)
        return ax

    genes_top = adata.var['r_time_zeroed'].nlargest(20).index
    genes_bot = adata.var['r_time_zeroed'].nsmallest(20).index
    fig, axs = plt.subplots(2, 3, figsize=(7, 4.5), sharex=True, sharey=True)
    for i, (ax, gene) in enumerate(zip(axs[0], genes_top)):
        plot_timecourse_gene(gene, ax=ax)
        if i != 0:
            ax.set_ylabel('')
    for i, (ax, gene) in enumerate(zip(axs[1], genes_bot)):
        plot_timecourse_gene(gene, ax=ax)
        if i != 0:
            ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig('../figures/figure2/top_bottom_genes_corr_time.svg')
    fig.savefig('../figures/figure2/top_bottom_genes_corr_time.png', dpi=600)
    plt.ion(); plt.show()

    print('Straightout DEGs between second half and first half')
    adata.obs['half'] = adata.obs['IncubationTime'].map({
        0: 'first', 5: 'first', 10: 'second', 20: 'second', 30: 'second',
    })
    adata1 = adata[adata.obs['half'] == 'first']
    adata2 = adata[adata.obs['half'] == 'second']
    res = anndataks.compare(adata1, adata2)

    def plot_deg(gene, ax=None, legend=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        x1 = adata1[:, gene].X.toarray()[:, 0] + 1
        x2 = adata2[:, gene].X.toarray()[:, 0] + 1
        ax.ecdf(x1, complementary=True, label='early', color='k')
        ax.ecdf(x2, complementary=True, label='late', color='tomato')
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlabel('Expression [cpm]')
        ax.set_title(gene)
        if legend:
            ax.legend()
        return ax

    top_degs = res.nlargest(40, 'statistic').index
    fig, axs = plt.subplots(2, 1, figsize=(2.5, 4.5), sharex=True, sharey=True)
    for i, (gene, ax) in enumerate(zip(['S100a11', 'Clint1'], axs.ravel())):
        plot_deg(gene, ax=ax, legend=bool(i))
        if i != 1:
            ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig('../figures/figure2/degs_early_late.svg')
    fig.savefig('../figures/figure2/degs_early_late.png', dpi=600)
    plt.ion(); plt.show()


