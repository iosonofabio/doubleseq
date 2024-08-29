# vim: fdm=indent
'''
author:     Fabio Zanini
date:       18/08/24
content:    Initial test to check out the double-seq data.
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

if __name__ == "__main__":

    pa = argparse.ArgumentParser()
    pa.add_argument('--plot', action='store_true')
    args = pa.parse_args()

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

    if args.plot:
        print('Plot DC vs T gene expression')
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

    if args.plot:
        print('Plot distributions of expression for markers of each cell type in the three groups')
        genes = [['Cd2', 'Cd6', 'Cd3d', 'Cd247'], ['Cd209a', 'Cd83', 'H2-Ab1', 'H2-DMa']]
        colors = {'T': 'gold', 'DC': 'deeppink', 'dou': '#333'}
        fig, axs = plt.subplots(4, 2, figsize=(5, 10), sharey=True)
        for genes_ct, ax_col in zip(genes, axs.T):
            for ir, (gene, ax) in enumerate(zip(genes_ct, ax_col)):
                for ct, adatai in adatad.items():
                    color = colors[ct]
                    x = adatai[:, gene].X.toarray().ravel()
                    ax.ecdf(x + 0.9, complementary=True, color=color, lw=2, label=ct)
                ax.text(0.9, 0.9, gene, ha='right', va='top', transform=ax.transAxes,
                        bbox=dict(facecolor='white', pad=5))
                ax.set_xlim(left=0.8)
                ax.set_xscale('log')
                ax.set_ylim(0, 0.6)
                ax.grid(True)
            if ir == len(genes[0]) - 1:
                ax.set_xlabel('Gene expression [cpm]')
        fig.tight_layout()
        plt.ion(); plt.show()

        print('Variation on the theme with kdes')
        from scipy.stats import gaussian_kde
        genes = [['Cd2', 'Cd6', 'Cd3d', 'Cd247'], ['Cd209a', 'Cd83', 'H2-Ab1', 'H2-DMa']]
        colors = {'T': 'gold', 'DC': 'mediumvioletred', 'dou': '#333'}
        xx = np.linspace(0, 16, 100)
        ysh = 0.03
        fig, axs = plt.subplots(4, 2, figsize=(5, 10), sharey=True)
        for genes_ct, ax_col in zip(genes, axs.T):
            for ir, (gene, ax) in enumerate(zip(genes_ct, ax_col)):
                for ict, (ct, adatai) in enumerate(adatad.items()):
                    color = colors[ct]
                    x = np.log1p(adatai[:, gene].X.toarray().ravel())
                    if x.max() == 0:
                        yy = xx * 0
                    elif (x > 0).sum() == 1:
                        val = x.max()
                        yy = 0.02 * gaussian_kde([val - 0.3, val + 0.3], bw_method=1)(xx)
                    else:
                        #yy = gaussian_kde(x, bw_method=1)(xx)
                        fac = (x > 0).mean()
                        yy = fac * gaussian_kde(x[x > 0], bw_method=1)(xx)
                    ax.fill_between(xx, (2 - ict) * ysh + yy, y2=(2 - ict) * ysh,
                                    facecolor=color, edgecolor='white', label=ct,
                                    lw=2,
                                    alpha=0.7, zorder=5 + 0.1 * ict)
                    ax.axhline((2 - ict) * ysh, color=color, lw=2, ls='--', zorder=5)
                    #ax.plot(x, y, drawstyle='steps-pre', color="white", lw=2)
                ax.text(0.1, 0.9, gene, ha='left', va='top', transform=ax.transAxes,
                        bbox=dict(facecolor='white', pad=5))
                ax.set_xlim(xx[0], xx[-1])
                #ax.set_xscale('log')
                #ax.set_ylim(0, 0.6)
                ax.grid(True)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        fig.text(0.5, 0.02, 'Gene expression [log1p(cpm)]', ha='center')
        ax = axs[0, 0]
        ax.text(15, ysh * 2.1, 'T', ha='right')
        ax.text(15, ysh * 1.1, 'DC', ha='right')
        ax.text(15, ysh * 0.1, 'doublets', ha='right')
        axs[0, 0].set_title('T cell markers')
        axs[0, 1].set_title('DC markers')
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig('../figures/fig2_distros_markers_T_DC.svg')
        fig.savefig('../figures/fig2_distros_markers_T_DC.png', dpi=600)
        plt.ion(); plt.show()

    print('Plot genes that are only expressed in doublets')
    df_frac = pd.DataFrame([], index=adata_dou.var_names)
    for ct,  adatai in adatad.items():
        df_frac[ct] =  np.asarray((adatai.X > 0).mean(axis=0))[0]

    genes_cands = df_frac.index[df_frac[['T', 'DC']].max(axis=1) == 0]
    genes_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=0))[0], index=genes_cands)
    genes_new = genes_new.loc[genes_new > 0]
    n_new = pd.Series(np.asarray((adata_dou[:, genes_cands].X > 0).sum(axis=1))[:, 0], index=adata_dou.obs_names)

    print('Check expression of genes with known roles in DC synapses')
    genes_rab = adata_dou.var_names[adata_dou.var_names.str.startswith('Rab')]
    df_rab = df_frac.loc[genes_rab]
    genes_snare_human = list(pd.read_csv('../data/literature/gene_groups/HGNC_group-1124_SNARE.csv', skiprows=1, header=None, index_col=1).iloc[1:].index)
    genes_snare_human += list(pd.read_csv('../data/literature/gene_groups/HGNC_group-818_syntaxins.csv', skiprows=1, header=None, index_col=1).iloc[1:].index)
    genes_snare_human += ['VAMP'+str(x) for x in (1, 2, 3, 4, 5, 7, 8, '9P')]
    genes_snare = list(m2h.loc[m2h.iloc[:, -1].isin(genes_snare_human)]['external_gene_name'].values)
    df_snare = df_frac.loc[genes_snare]
    genes_crosspresent = ['Tap1', 'Cybb', 'Rac1'] #NOTE: Nox2 = Cybb
    genes_translocon = adata_dou.var_names[adata_dou.var_names.str.startswith('Sec61')]
    genes_select = list(genes_rab) +  list(genes_snare) + list(genes_crosspresent) + list(genes_translocon)
    df_select = df_frac.loc[genes_select]
    df_select = df_select.loc[df_select.max(axis=1) > 0]
    if args.plot:
        Z = linkage((df_select.T / df_select.sum(axis=1)).T.values, method='average', optimal_ordering=True)
        sns.clustermap(
            (df_select.T / df_select.sum(axis=1)).T,
            row_linkage=Z,
            figsize=(3, 20),
            yticklabels=True,
        )
        plt.ion(); plt.show()

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

    if args.plot:
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

    print('Check how much overlap there is between cells expressing these new genes')
    fig, ax = plt.subplots(figsize=(3.3, 3))
    ax.ecdf(n_new.values, complementary=True, lw=2, color='tomato')
    ax.set_xlabel('# new genes expressed')
    ax.set_ylabel('fraction of doublets with\n>x new genes')
    ax.grid(True)
    ax.axhline(0.16, color='#333', ls='--', lw=2, alpha=0.9)
    ax.text(750, 0.05, '14', clip_on=False)
    ax.text(750, 0.50, '72', clip_on=False)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig('../figures/fig2_n_new_genes.svg')
    fig.savefig('../figures/fig2_n_new_genes.png', dpi=600)
    plt.ion(); plt.show()

    print('Check if doublets with lots of new genes are later on')
    cells_many_new = n_new.nlargest(14).index
    cells_other = n_new.sort_values(ascending=False).iloc[14:].index
    ts_many_new = adata_dou[cells_many_new].obs['Timepoint(min)']
    ts_other = adata_dou[cells_other].obs['Timepoint(min)']
    from scipy.stats import ks_2samp
    res = ks_2samp(ts_many_new, ts_other)
    
    fig, ax = plt.subplots()
    ax.ecdf(ts_many_new, complementary=True, color='tomato', label='Top new gene expressors')
    ax.ecdf(ts_other, complementary=True, color='#333', label='Other doublets')
    ax.grid(True)
    ax.set_xlabel('Time [min]')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.ion(); plt.show()

    print('Check if the same doublets have more reads/genes in general')
    adata_dou.obs['ngenes'] = (adata_dou.X > 0).sum(axis=1)
    nreads_many_new = adata_dou[cells_many_new].obs['total']
    nreads_other = adata_dou[cells_other].obs['total']
    from scipy.stats import ks_2samp
    res = ks_2samp(nreads_many_new, nreads_other)
    ngenes_many_new = adata_dou[cells_many_new].obs['ngenes']
    ngenes_other = adata_dou[cells_other].obs['ngenes']
    resg = ks_2samp(ngenes_many_new, ngenes_other)

    pX_mg = np.asarray(adata_dou[cells_many_new].X.todense()) / 1e6
    pX_other = np.asarray(adata_dou[cells_other].X.todense()) / 1e6
    entropy_many_new = -pd.Series((pX_mg * np.log2(pX_mg + 1e-9)).sum(axis=1), index=cells_many_new)
    entropy_other = -pd.Series((pX_other * np.log2(pX_other + 1e-9)).sum(axis=1), index=cells_other)
    resS = ks_2samp(entropy_many_new, entropy_other)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    ax = axs[0]
    ax.ecdf(nreads_many_new, complementary=True, lw=2, color='tomato', label='Top new gene expressors')
    ax.ecdf(nreads_other, complementary=True, lw=2, color='#333', label='Other doublets')
    ax.text(0.9, 0.9, '$P_{KS} = 0.01$', transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
    ax.grid(True)
    ax.set_xlabel('Number of reads')
    ax.set_xlim(left=0)

    ax = axs[1]
    ax.ecdf(ngenes_many_new, complementary=True, lw=2, color='tomato', label='Top expressors\nof new genes')
    ax.ecdf(ngenes_other, complementary=True, lw=2, color='#333', label='Other\ndoublets')
    ax.text(0.9, 0.9, '$P_{KS} = 0.02$', transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
    ax.grid(True)
    ax.set_xlabel('Number of expressed genes')
    ax.set_xlim(left=0)

    ax = axs[2]
    ax.ecdf(entropy_many_new, complementary=True, lw=2, color='tomato', label='Top expressors\nof new genes')
    ax.ecdf(entropy_other, complementary=True, lw=2, color='#333', label='Other\ndoublets')
    ax.grid(True)
    ax.set_xlabel('Entropy [bits]')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig('../figures/fig2_ks_doublets_new_genes.svg')
    fig.savefig('../figures/fig2_ks_doublets_new_genes.png', dpi=600)
    plt.ion(); plt.show()

    print('DEG between those two kinds of doublets')
    adata_dou_mg = adata_dou[cells_many_new]
    adata_dou_other = adata_dou[cells_other]
    res = anndataks.compare(adata_dou_mg, adata_dou_other)
    genes = ['Hnrnpk', 'Sfi1', 'Shisa5']
    for gene in genes:
        fig, ax = plt.subplots(figsize=(5, 3))
        tmp_mg = np.asarray(adata_dou_mg[:, gene].X.todense())[:, 0]
        tmp_other = np.asarray(adata_dou_other[:, gene].X.todense())[:, 0]
        ax.ecdf(tmp_mg + 1, complementary=True, lw=2, color='tomato', label='Top expressors\nof new genes')
        ax.ecdf(tmp_other + 1, complementary=True, lw=2, color='#333', label='Other\ndoublets')
        ax.set_xlim(left=0.9)
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlabel(f'Expression of {gene}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        fig.tight_layout()
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
