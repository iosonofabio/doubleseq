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
import seaborn as sns
import anndata
import anndataks
import scanpy as sc
import gseapy as gp

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
    df_quads = [
        df.loc[(df['x'] > 0) & (df['y'] > 0)],
        df.loc[(df['x'] < 0) & (df['y'] > 0)],
        df.loc[(df['x'] < 0) & (df['y'] < 0)],
        df.loc[(df['x'] > 0) & (df['y'] < 0)],
    ]
    for dfi in df_quads:
        for gene, row in dfi.nlargest(20, 'r').iterrows():
            xi = row['x']
            yi = row['y']
            text = ax.text(xi, yi, gene, ha='center', va='bottom')
            texts.append(text)
    adjust_text(texts)
    fig.tight_layout()
    plt.ion(); plt.show()

    print('Gene set enrichment analysis')
    df['angle'] = (np.arctan2(df['y'], df['x']) * 180 / np.pi + 360) % 360
    genes_T = df.index[(df['r'] > 0.1) & (df['angle'] > 60) & (df['angle'] < 120)]
    genes_DC = df.index[(df['r'] > 0.1) & (df['angle'] < 30) | (df['angle'] > 300)]
    genes_dou = df.index[((df['r'] > 0.15) & (df['angle'] > 30) & (df['angle'] < 60)) | ((df['r'] > 0.1) & (df['angle'] > 210) & (df['angle'] < 240))]

    # if you are only intrested in dataframe that enrichr returned, please set outdir=None
    enr_T = gp.enrichr(gene_list=list(genes_T),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    enr_T.results.iloc[:5]

    enr_DC = gp.enrichr(gene_list=list(genes_DC),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    enr_DC.results.iloc[:15]

    enr_dou = gp.enrichr(gene_list=list(genes_dou),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    enr_dou.results.iloc[:25]

    print('Look very strictly, at genes found higher than in any T or DC')
    emax_T = pd.Series(np.asarray(adata_T.X.max(axis=0).todense())[0], index=adata_T.var_names, name='T')
    emax_DC = pd.Series(np.asarray(adata_DC.X.max(axis=0).todense())[0], index=adata_DC.var_names, name='DC')
    emax_dou = pd.Series(np.asarray(adata_dou.X.max(axis=0).todense())[0], index=adata_dou.var_names, name='double')
    emax = pd.concat([emax_T, emax_DC, emax_dou], axis=1)
    frac_peeking = pd.Series(np.asarray((adata_dou.X > emax[['T', 'DC']].max(axis=1).values).mean(axis=0))[0], index=adata_dou.var_names)
    emax['frac_double_peeking'] = frac_peeking
    emax['frac_T'] = de_vs_T['frac2']
    emax['frac_DC'] = de_vs_DC['frac2']
    emax['frac_dou'] = de_vs_T['frac1']
    emax['avg_T'] = de_vs_T['avg2']
    emax['avg_DC'] = de_vs_DC['avg2']
    emax['avg_dou'] = de_vs_T['avg1']


    print('Correlate gene expression with time in doublets')
    from scipy.stats import pearsonr, spearmanr
    time_dou = adata_dou.obs['Timepoint(min)'].values
    mat = np.asarray(adata_dou.X.todense())
    rs, rhos = [], []
    for i, row in enumerate(mat.T):
        if i % 1000 == 0:
            print(i + 1)
        r = pearsonr(time_dou, row)
        rho = spearmanr(time_dou, row)
        rs.append(list(r))
        rhos.append(list(rho))
    rs = pd.DataFrame(rs, columns=['r', 'r_pvalue'])
    rhos = pd.DataFrame(rhos, columns=['rho', 'rho_pvalue'])
    corrs = pd.concat([rs, rhos], axis=1)
    corrs.index = adata_dou.var_names

    genes_top_rho = corrs.nlargest(100, 'rho').index
    enr_top_rho = gp.enrichr(gene_list=list(genes_top_rho),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    genes_bot_rho = corrs.nsmallest(100, 'rho').index
    enr_bot_rho = gp.enrichr(gene_list=list(genes_bot_rho),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )
    enr_rho = gp.enrichr(gene_list=list(genes_bot_rho) + list(genes_top_rho),
                 gene_sets=['KEGG_2019_Mouse'],
                 organism='mouse',
                 outdir=None,
                )

    print('Plot a few genes over time to get a sense of what it is')
    bins_no = np.linspace(time_dou.min(), time_dou.max(), 12)
    bins = [[bins_no[i], bins_no[i + 2]] for i in range(len(bins_no) - 2)]
    xs = [np.mean(x) for x in bins]
    genes = list(corrs['rho'].nlargest(6).index)
    genes += list(corrs['rho'].nsmallest(6).index)
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
    axs = axs.ravel()
    for gene, ax in zip(genes, axs):
        means = []
        fracs = []
        quartiles = []
        for ib, (bs, be) in enumerate(bins):
            idx = (time_dou >= bs) & (time_dou <= be)
            datum = np.asarray(adata_dou[adata_dou.obs_names[idx], gene].X.todense()).T[0]
            xi = xs[ib]
            means.append(datum.mean())
            fracs.append((datum > 0).mean())
            quartile = np.percentile(datum, [10, 25, 50, 75, 90])
            quartiles.append(quartile)
        means = np.array(means)
        quartiles = np.asarray(quartiles).T
        ax.plot(xs, np.maximum(means, 1), color='r', marker='+')
        ax2 = ax.twinx()
        ax2.plot(xs, np.maximum(fracs, 0.01), color='steelblue', marker='*')
        ax2.set_ylim([0.009, 1.01])
        ax.bar(xs, quartiles[3], width=[(x[1] - x[0]) / 2 for x in bins], bottom=quartiles[1], ec='k', facecolor='grey', alpha=0.7)
        ax.errorbar(xs, quartiles[2], yerr=np.array([quartiles[2] - quartiles[0], quartiles[4] - quartiles[2]]), color='k')
        ax.set_xlabel('Time [mins]')
        ax.set_ylabel('Expression [cpm]')
        ax.grid(True)
        ax.set_title(gene)
    fig.tight_layout()
    plt.ion(); plt.show()

    print('Correlate genes to one another')
    adata_dou.var['number'] = np.arange(adata_dou.n_vars)
    genes_tgt = ['Pf4', 'Dnajb1']
    rowigd = {gene: mat.T[adata_dou.var.at[gene, 'number']] for gene in genes_tgt}
    rhos_gened = {gene: [] for gene in genes_tgt}
    for i, row in enumerate(mat.T):
        if i % 1000 == 0:
            print(i + 1)
        for gene in genes_tgt:
            rho = spearmanr(rowigd[gene], row)
            rhos_gened[gene].append(list(rho))
    rhos_genem = [pd.DataFrame(rhos_gened[gene], columns=[f'rho_{gene}', f'rho_pvalue_{gene}']) for gene in genes_tgt]
    rhos_genem = pd.concat(rhos_genem, axis=1)
    rhos_genem.index = adata_dou.var_names

    print('Plot a few genes over time to get a sense of what it is')
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
    for igt, (genei, axrow) in enumerate(zip(genes_tgt, axs)):
        genes = list(rhos_genem[f'rho_{genei}'].nlargest(6).index)
        axrow[0].set_ylabel('Expression [cpm]')
        for gene, ax in zip(genes, axrow):
            means = []
            fracs = []
            quartiles = []
            for ib, (bs, be) in enumerate(bins):
                idx = (time_dou >= bs) & (time_dou <= be)
                datum = np.asarray(adata_dou[adata_dou.obs_names[idx], gene].X.todense()).T[0]
                xi = xs[ib]
                means.append(datum.mean())
                fracs.append((datum > 0).mean())
                quartile = np.percentile(datum, [10, 25, 50, 75, 90])
                quartiles.append(quartile)
            means = np.array(means)
            quartiles = np.asarray(quartiles).T
            ax.plot(xs, np.maximum(means, 1), color='r', marker='+')
            ax2 = ax.twinx()
            ax2.plot(xs, np.maximum(fracs, 0.01), color='steelblue', marker='*')
            ax2.set_ylim([0.009, 1.01])
            ax.bar(xs, quartiles[3], width=[(x[1] - x[0]) / 2 for x in bins], bottom=quartiles[1], ec='k', facecolor='grey', alpha=0.7)
            ax.errorbar(xs, quartiles[2], yerr=np.array([quartiles[2] - quartiles[0], quartiles[4] - quartiles[2]]), color='k')
            ax.grid(True)
            ax.set_title(gene)
            if igt == 1:
                ax.set_xlabel('Time [mins]')
    fig.tight_layout()
    plt.ion(); plt.show()

    print('Corelate gene expression and time in non-doublets, as controls')
    time_DC = adata_DC.obs['Timepoint(min)'].values
    mat = np.asarray(adata_DC.X.todense())
    rs, rhos = [], []
    for i, row in enumerate(mat.T):
        if i % 1000 == 0:
            print(i + 1)
        r = pearsonr(time_DC, row)
        rho = spearmanr(time_DC, row)
        rs.append(list(r))
        rhos.append(list(rho))
    rs = pd.DataFrame(rs, columns=['r', 'r_pvalue'])
    rhos = pd.DataFrame(rhos, columns=['rho', 'rho_pvalue'])
    corrs_DC = pd.concat([rs, rhos], axis=1)
    corrs_DC.index = adata_DC.var_names

    time_T = adata_T.obs['Timepoint(min)'].values
    mat = np.asarray(adata_T.X.todense())
    rs, rhos = [], []
    for i, row in enumerate(mat.T):
        if i % 1000 == 0:
            print(i + 1)
        r = pearsonr(time_T, row)
        rho = spearmanr(time_T, row)
        rs.append(list(r))
        rhos.append(list(rho))
    rs = pd.DataFrame(rs, columns=['r', 'r_pvalue'])
    rhos = pd.DataFrame(rhos, columns=['rho', 'rho_pvalue'])
    corrs_T = pd.concat([rs, rhos], axis=1)
    corrs_T.index = adata_T.var_names

    corrsd = {'dou': corrs, 'DC': corrs_DC, 'T': corrs_T}

    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for ct, ax in zip(['DC', 'T'], axs):
        texts = []
        x = corrsd[ct]['rho']
        y = corrsd['dou']['rho']
        r = np.sqrt(x**2 + y**2)
        ax.scatter(x, y, color='k', alpha=0.5)
        sns.kdeplot(x=x, y=y, ax=ax, fill=False)
        genes = r.nlargest(30).index
        for gene in genes:
            txt = ax.text(x.at[gene], y.at[gene], gene, ha='center', va='bottom')
            texts.append(txt)
        ax.grid(True)
        ax.set_xlabel(f'$\\rho ({ct})$')
        #adjust_text(texts)
    axs[0].set_ylabel('$\\rho (doublets)$')
    fig.tight_layout()
    plt.ion(); plt.show()
