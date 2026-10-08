import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pyranges as pr
from scipy.stats import spearmanr
from tqdm import tqdm
import pybedtools
import pysam
import pychromvar as pc
from pyfaidx import Fasta
import pyranges as pr
from pyjaspar import jaspardb

'''the analyses needs the atac python environment'''

def generate_peak_insertion(atac_adata, cutoff):
    fragments_array = atac_adata.X.copy()
    fragments_array_final = np.array(fragments_array)
    fragments_array_final_insertion = np.maximum(0, np.ceil(fragments_array_final - cutoff)).astype(
        int)  # np.maximum(0, fragments_array_final)#0.05

    return fragments_array_final_insertion

def pseudo_coverage_plot(gene_name, atac_adata, gtf_dir, cell_label, lower_y=-0.01, upper_y=1.0):

    gtf = pr.read_gtf(gtf_dir)

    genes = gtf[gtf.Feature == "gene"]
    genes_start = np.array(genes.Start)
    genes_end = np.array(genes.End)
    genes_strand = np.array(genes.Strand)
    TSS = [x if z == '+' else y for (x, y, z) in zip(genes_start, genes_end, genes_strand)]
    genes.TSS = TSS

    ### build promoters
    promoters = genes.copy()
    promoters.Start = genes.TSS - 2000
    promoters.End = genes.TSS + 200
    promoters.Name = promoters.gene_name

    peak_df = atac_adata.var_names.to_series().str.split("[:-]", expand=True)
    peak_df.columns = ["Chromosome", "Start", "End"]
    peak_df["Start"] = peak_df["Start"].astype(int)
    peak_df["End"] = peak_df["End"].astype(int)
    peak_df['Index'] = peak_df.index

    peaks = pr.PyRanges(peak_df)

    region = genes[genes.gene_name == gene_name]

    peaks_sub = peaks.overlap(region)
    peak_idx = peaks_sub.df.Index.values

    if len(peak_idx) == 0:
        raise ValueError(f"No peaks are overlapping with the region of gene {gene_name}.")
    else:
        peak_mat = atac_adata[:, peak_idx].X.toarray()
        groups = atac_adata.obs[cell_label]
        group_names = groups.unique()

        group_signal = {}

        for g in group_names:
            idx = np.where(groups == g)[0]
            group_signal[g] = peak_mat[idx, :].mean(axis=0)

            plot_nrow = len(np.unique(atac_adata.obs[cell_label]))

            fig, axs = plt.subplots(plot_nrow, 1, figsize=(8, plot_nrow*2))

            for g in range(plot_nrow):
                lower_y = lower_y
                upper_y = upper_y

                # Plot each group's pseudo coverage
                xs = peaks_sub.df.Start.values
                xe = peaks_sub.df.End.values
                ys = group_signal[group_names[g]]

                # plot rectangles
                for x1, x2, h in zip(xs, xe, ys):
                    axs[g].fill_between([x1, x2], 0, h, alpha=0.5, label=g)

                # Gene bodies overlapping region
                genes_sub = region.df  # genes.overlap(region).df

                for _, row in genes_sub.iterrows():
                    axs[g].hlines(
                        y=upper_y / 40,
                        xmin=row.Start,
                        xmax=row.End,
                        color='black',
                        linewidth=3
                    )
                    axs[g].text(
                        (row.Start + row.End) / 2,
                        lower_y / 5,  # 0.1,#-0.15,
                        row.gene_name,
                        ha="center",
                        fontsize=8
                    )

                # Promoters
                # prom_sub = promoters.overlap(region).df
                prom_sub = promoters[promoters.Name == gene_name].df

                for _, row in prom_sub.iterrows():
                    axs[g].add_patch(plt.Rectangle(
                        (row.Start, upper_y / 40),  # -0.05
                        row.End - row.Start,
                        upper_y / 40,
                        facecolor="firebrick",
                        alpha=0.7
                    ))

                axs[g].set_ylim(lower_y, upper_y)
                axs[g].set_xlabel(
                    f"Genomic position: {np.array(genes_sub)[0][0]}: {np.array(genes_sub)[0][3]} - {np.array(genes_sub)[0][4]}")
                axs[g].set_ylabel("Accessibility")
                axs[g].set_title(f"Pseudo-CoveragePlot of gene {gene_name} in cluster {group_names[g]}")

            plt.tight_layout()
            plt.show()

    return group_signal

def compute_peak_gene_links(atac, rna, gene_coords,
                            max_dist=500000,
                            min_rho=0.2,
                            max_p=1e-6):
    """
    Compute peak-gene links for paired Multiome data.

    Parameters
    ----------
    atac : AnnData
        scATAC peak x cell matrix.
    rna : AnnData
        scRNA gene x cell matrix.
    gene_coords : dict
        {gene_name: (chrom, tss_position)} dictionary.
    max_dist : int
        maximum distance from peak to gene TSS.
    min_rho : float
        minimum Spearman correlation.
    max_p : float
        maximum p-value for significance.

    Returns
    -------
    DataFrame with columns:
    peak, gene, rho, pval, dist
    """

    peak_coords = [p.replace(":", "-").split("-") for p in atac.var_names]
    peak_positions = [
        (chrom, int(start), int(end)) for chrom, start, end in peak_coords
    ]

    results = []

    atac_matrix = atac.X.toarray()
    rna_matrix = rna.X.copy() #.toarray()

    print("Computing peak–gene correlations...")

    for gi, gene in enumerate(tqdm(rna.var_names)):
        if gene not in gene_coords:
            continue
        chrom_g, pos_g = gene_coords[gene]

        expr = rna_matrix[:, gi]

        for pi, peak in enumerate(atac.var_names):
            chrom_p, start_p, end_p = peak_positions[pi]
            if chrom_p != chrom_g:
                continue

            # TSS distance
            d = min(abs(start_p - pos_g), abs(end_p - pos_g))
            if d > max_dist:
                continue

            acc = atac_matrix[:, pi]

            rho, p = spearmanr(acc, expr)
            if rho >= min_rho and p <= max_p:
                results.append([peak, gene, rho, p, d])

    return pd.DataFrame(results,
        columns=["peak", "gene", "rho", "pval", "dist"])


def gene_coords(gtf_dir):
    gtf = pr.read_gtf(gtf_dir)

    genes = gtf[gtf.Feature == "gene"]

    genes_start = np.array(genes.Start)
    genes_end = np.array(genes.End)
    genes_strand = np.array(genes.Strand)
    TSS = [x if z == '+' else y for (x, y, z) in zip(genes_start, genes_end, genes_strand)]
    genes.TSS = TSS

    gene_coords = {}
    genes_array = np.array(genes.gene_name)
    chr_array = np.array(genes.Chromosome)
    tss_array = np.array(genes.TSS)
    for i in range(len(genes_array)):
        gene_coords[genes_array[i]] = (chr_array[i], tss_array[i])

    return gene_coords

def pychromvar_motif(adata, fasta_dir, gtf_dir):

    """
    :param adata: anndata, adata.X should be the predicted peak matrix processed with generate_peak_insertion
    :param fasta_dir: string, directory of hg38.fa file
    :param gtf_dir: string, directory of gencode.v38.annotation.gtf file
    :return: dev: anndata with deviation score for motifs, adata: processed anndata with motif hit matrix included
    """

    peak_df = adata.var.index.to_series().str.split(':|-', expand=True)
    peak_df.columns = ["Chromosome", "Start", "End"]
    peak_df[["Start", "End"]] = peak_df[["Start", "End"]].astype(int)
    adata.var = adata.var.assign(**peak_df)

    fasta = Fasta(fasta_dir)
    gtf = pr.read_gtf(gtf_dir)
    genes = gtf[gtf.Feature == "gene"]

    tss = genes.apply(
        lambda df: df.assign(
            TSS=df.Start.where(df.Strand == '+', df.End)
        )
    )[['Chromosome', 'TSS', 'gene_id', 'gene_name', 'Strand']]

    peaks = pr.PyRanges(adata.var[['Chromosome', 'Start', 'End']])

    promoters = genes.slack(2000).subset(lambda r: r.Feature == "gene")

    annot = peaks.join(promoters)

    nearest = peaks.nearest(genes)
    adata.var['nearest_gene'] = nearest.gene_name.values

    peak_names = [p.replace(":", "-") for p in adata.var_names]
    adata.var_names = peak_names

    pc.add_peak_seq(adata, genome_file=fasta_dir)
    pc.add_gc_bias(adata)
    pc.get_bg_peaks(adata)

    jdb_obj = jaspardb(release='JASPAR2024')
    motifs = jdb_obj.fetch_motifs(
        collection='CORE',
        tax_group=['vertebrates'])

    pc.match_motif(adata, motifs=motifs, p_value=5e-05)

    adata.X = adata.X.astype('float64')

    dev = pc.compute_deviations(adata)

    return dev, adata


def integrate_layers(
    peak_gene_df,
    motif_hits_df,
    tf_activity_df=None,
    tf_expression_df=None,
    dist_scale=50000,
    min_score=0.01,
    use_motif_scores=False
):
    """
    Build an integrated TF–enhancer–gene GRN using:
    - peak→gene statistical links
    - TF motif hits per peak (DataFrame)
    - optional TF activity (chromVAR / pychromvar)
    - optional TF expression
    - ABC-like distance weighting

    Parameters
    ----------
    peak_gene_df : DataFrame
        Columns: peak, gene, rho, pval, dist
    motif_hits_df : DataFrame (TF × peak)
        TF rows, peak columns.
        Binary (0/1) OR motif scores.
    tf_activity_df : DataFrame (optional)
        Rows = TF, columns = cells (chromVAR Z-scores).
    tf_expression_df : DataFrame (optional)
        Rows = TF, columns = cells.
    dist_scale : int
        Distance decay scale for ABC-like weighting.
    min_score : float
        Minimum GRN score to keep.
    use_motif_scores : bool
        If True, uses numeric motif scores from motif_hits_df.
        If False, any nonzero = 1.

    Returns
    -------
    grn_df : DataFrame
        Columns: TF, Gene, Peak, Score, MotifScore,
                 PeakGeneRho, DistanceWeight, TFsupport
    """

    peaks = motif_hits_df.columns
    TFs = motif_hits_df.index

    grn_records = []

    for _, row in peak_gene_df.iterrows():
        peak = row["peak"]
        gene = row["gene"]
        rho_pg = row["rho"]
        dist = row["dist"]

        if peak not in peaks:
            continue

        # ABC-like distance penalty
        dist_w = 1 / (1 + (dist / dist_scale)**2)

        # all TFs that hit this peak (binary or score-based)
        for tf in TFs:
            motif_val = motif_hits_df.loc[tf, peak]

            # Skip non-hits if binary
            if not use_motif_scores and motif_val <= 0:
                continue

            # If numeric scores, motif_val is the PWM score
            motif_score = float(motif_val)

            # Compute TF support score
            tf_support = 1.0

            # TF expression support (if provided)
            if tf_expression_df is not None and tf in tf_expression_df.index:
                # mean or variance could be used; here mean activity
                tf_exp_val = float(tf_expression_df.loc[tf].mean())
                tf_support *= max(tf_exp_val, 0.01)

            # TF chromVAR activity support
            if tf_activity_df is not None and tf in tf_activity_df.index:
                tf_act_val = float(tf_activity_df.loc[tf].mean())
                tf_support *= max(tf_act_val, 0.01)

            # Final integrated GRN score
            score = rho_pg * motif_score * dist_w * tf_support

            if score >= min_score:
                grn_records.append([
                    tf,
                    gene,
                    peak,
                    score,
                    motif_score,
                    rho_pg,
                    dist_w,
                    tf_support
                ])

    return pd.DataFrame(
        grn_records,
        columns=[
            "TF", "Gene", "Peak", "Score",
            "MotifScore", "PeakGeneRho",
            "DistanceWeight", "TFsupport"
        ]
    ).sort_values("Score", ascending=False)










