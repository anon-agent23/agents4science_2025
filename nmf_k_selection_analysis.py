#!/usr/bin/env python3
"""
NMF K-Selection Optimization Algorithm
=====================================

This module implements multiple methods for optimal k-selection in Non-negative Matrix 
Factorization (NMF) for transcriptomic data analysis.

The primary goal is to maximize discrimination of gene program usage scores between 
sample groups, with focus on bimodal distributions highly correlated with group factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from pydeseq2.dds import DeseqDataSet
import warnings
warnings.filterwarnings('ignore')

class NMFKSelector:
    """
    Advanced k-selection algorithm for NMF transcriptomic analysis.
    
    Implements multiple metrics for optimal k selection:
    1. Group Correlation Metric (primary criterion)
    2. Bimodality Metric (secondary criterion) 
    3. Silhouette Analysis
    4. Stability Analysis
    5. Reconstruction Error Analysis
    """
    
    def __init__(self, random_state=42, max_iter=5000, n_init=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.results_ = {}
        
    def preprocess_data(self, counts, metadata, n_variable_genes=5000):
        """
        Preprocess transcriptomic data following standard pipeline.
        
        Parameters:
        -----------
        counts : pd.DataFrame
            Raw gene count matrix (genes x samples)
        metadata : pd.DataFrame  
            Sample metadata with 'Group' column
        n_variable_genes : int
            Number of highly variable genes to select
            
        Returns:
        --------
        df_expr : pd.DataFrame
            Preprocessed expression matrix
        groups : pd.Series
            Sample group labels
        """
        print("Starting data preprocessing...")
        
        # Transpose to samples x genes
        counts_t = counts.T
        
        # Filter low-count genes
        counts_filtered = counts[counts.sum(axis=1) > 10]
        counts_t = counts_filtered.T
        
        # DESeq2 normalization
        print("Performing DESeq2 normalization...")
        # Ensure sample names match between counts and metadata
        counts_t = counts_t.loc[metadata.index]
        dds = DeseqDataSet(counts=counts_t, metadata=metadata, design="~Group")
        dds.deseq2()
        
        # Extract normalized counts
        norm_counts = pd.DataFrame(
            dds.layers['normed_counts'], 
            index=dds.obs.index, 
            columns=dds.var.index
        )
        
        # Select highly variable genes
        print(f"Selecting {n_variable_genes} highly variable genes...")
        gene_means = norm_counts.mean(axis=0)
        gene_vars = norm_counts.var(axis=0)
        
        # Remove zero-mean genes
        nonzero_mask = gene_means > 0
        mu = gene_means[nonzero_mask].values
        var = gene_vars[nonzero_mask].values
        genes_nonzero = norm_counts.columns[nonzero_mask]
        
        # Fit variance-mean relationship
        coeff = np.polyfit(mu, var, deg=2)
        var_est = coeff[0]*mu*mu + coeff[1]*mu + coeff[2]
        vscore = (var - var_est) / (mu + 1e-9)
        
        # Select top variable genes
        ranked_idx = np.argsort(-vscore)
        top_genes = genes_nonzero[ranked_idx[:n_variable_genes]]
        df_expr = norm_counts[top_genes]
        
        # Log2 transformation
        df_expr = np.log2(df_expr + 1)
        
        # Get group labels
        groups = metadata['Group']
        
        print(f"Preprocessing complete. Matrix shape: {df_expr.shape}")
        return df_expr, groups
    
    def group_correlation_metric(self, W, groups):
        """
        Calculate correlation between NMF factors and group membership.
        
        This is the PRIMARY criterion for k-selection.
        """
        group_encoded = pd.get_dummies(groups)
        correlations = []
        
        for i in range(W.shape[1]):
            factor_scores = W[:, i]
            max_corr = 0
            
            # Find maximum correlation with any group
            for group_col in group_encoded.columns:
                group_vector = group_encoded[group_col].values
                corr, _ = pearsonr(factor_scores, group_vector)
                max_corr = max(max_corr, abs(corr))
            
            correlations.append(max_corr)
        
        # Return mean of top correlations (focuses on discriminative factors)
        return np.mean(sorted(correlations, reverse=True)[:min(3, len(correlations))])
    
    def bimodality_metric(self, W, groups):
        """
        Measure bimodality of usage scores - SECONDARY criterion.
        
        Good factors should show high usage in some groups, low in others.
        """
        bimodality_scores = []
        
        for i in range(W.shape[1]):
            factor_scores = W[:, i]
            
            # Calculate Hartigan's dip test statistic (approximation)
            # Higher values indicate more bimodal distributions
            sorted_scores = np.sort(factor_scores)
            n = len(sorted_scores)
            
            # Simple bimodality measure: kurtosis-based
            # Bimodal distributions have negative excess kurtosis
            kurt = stats.kurtosis(factor_scores)
            bimodality_score = -kurt + 1  # Transform so higher = more bimodal
            
            bimodality_scores.append(max(0, bimodality_score))
        
        return np.mean(bimodality_scores)
    
    def silhouette_analysis(self, W, groups):
        """
        Silhouette analysis for clustering quality.
        """
        try:
            group_labels = pd.Categorical(groups).codes
            silhouette_avg = silhouette_score(W, group_labels)
            return silhouette_avg
        except:
            return 0.0
    
    def stability_analysis(self, X, k, n_runs=5):
        """
        Measure stability of NMF decomposition across multiple runs.
        """
        all_W = []
        
        for run in range(n_runs):
            model = NMF(
                n_components=k, 
                init='random', 
                random_state=self.random_state + run,
                max_iter=self.max_iter
            )
            W = model.fit_transform(X)
            all_W.append(W)
        
        # Calculate pairwise correlations between runs
        correlations = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                # Match factors by maximum correlation
                W1, W2 = all_W[i], all_W[j]
                factor_corrs = []
                
                for f1 in range(k):
                    max_corr = 0
                    for f2 in range(k):
                        corr, _ = pearsonr(W1[:, f1], W2[:, f2])
                        max_corr = max(max_corr, abs(corr))
                    factor_corrs.append(max_corr)
                
                correlations.append(np.mean(factor_corrs))
        
        return np.mean(correlations)
    
    def reconstruction_error_analysis(self, X, k):
        """
        Analyze reconstruction error for given k.
        """
        model = NMF(
            n_components=k,
            init='random',
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        model.fit(X)
        return model.reconstruction_err_
    
    def composite_score(self, group_corr, bimodality, silhouette, stability, 
                       weights=(0.4, 0.25, 0.2, 0.15)):
        """
        Calculate composite score for k-selection.
        
        Weights prioritize group correlation (primary) and bimodality (secondary).
        """
        return (weights[0] * group_corr + 
                weights[1] * bimodality + 
                weights[2] * silhouette + 
                weights[3] * stability)
    
    def evaluate_k_range(self, X, groups, k_range=None):
        """
        Evaluate NMF performance across range of k values.
        """
        if k_range is None:
            k_range = range(2, min(15, X.shape[0]//2))
        
        print(f"Evaluating k values: {list(k_range)}")
        
        results = {
            'k': [],
            'group_correlation': [],
            'bimodality': [],
            'silhouette': [],
            'stability': [],
            'reconstruction_error': [],
            'composite_score': []
        }
        
        for k in k_range:
            print(f"Evaluating k={k}...")
            
            # Fit NMF
            model = NMF(
                n_components=k,
                init='random', 
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            W = model.fit_transform(X)
            
            # Calculate metrics
            group_corr = self.group_correlation_metric(W, groups)
            bimodality = self.bimodality_metric(W, groups)
            silhouette = self.silhouette_analysis(W, groups)
            stability = self.stability_analysis(X, k, n_runs=3)  # Reduced for speed
            recon_error = model.reconstruction_err_
            
            # Normalize metrics for composite score
            composite = self.composite_score(
                group_corr, 
                min(1.0, bimodality/2.0),  # Normalize bimodality 
                (silhouette + 1) / 2,      # Normalize silhouette to [0,1]
                stability
            )
            
            # Store results
            results['k'].append(k)
            results['group_correlation'].append(group_corr)
            results['bimodality'].append(bimodality)
            results['silhouette'].append(silhouette)
            results['stability'].append(stability)
            results['reconstruction_error'].append(recon_error)
            results['composite_score'].append(composite)
            
            print(f"  Group correlation: {group_corr:.3f}")
            print(f"  Bimodality: {bimodality:.3f}")
            print(f"  Composite score: {composite:.3f}")
        
        self.results_ = pd.DataFrame(results)
        return self.results_
    
    def select_optimal_k(self, results=None):
        """
        Select optimal k based on composite score and additional criteria.
        """
        if results is None:
            results = self.results_
            
        # Primary selection: highest composite score
        best_idx = results['composite_score'].idxmax()
        optimal_k = results.loc[best_idx, 'k']
        
        # Secondary validation: ensure group correlation is high
        min_group_corr = 0.3  # Threshold for meaningful group discrimination
        valid_k = results[results['group_correlation'] >= min_group_corr]
        
        if len(valid_k) > 0:
            # Among valid k values, select highest composite score
            best_valid_idx = valid_k['composite_score'].idxmax()
            optimal_k = valid_k.loc[best_valid_idx, 'k']
        
        print(f"\nOptimal k selected: {optimal_k}")
        print(f"Composite score: {results.loc[results['k']==optimal_k, 'composite_score'].iloc[0]:.3f}")
        print(f"Group correlation: {results.loc[results['k']==optimal_k, 'group_correlation'].iloc[0]:.3f}")
        
        return int(optimal_k)
    
    def plot_results(self, results=None, figsize=(15, 10)):
        """
        Visualize k-selection results.
        """
        if results is None:
            results = self.results_
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('NMF K-Selection Analysis Results', fontsize=16)
        
        k_values = results['k']
        
        # Plot individual metrics
        metrics = [
            ('group_correlation', 'Group Correlation', 'Primary Criterion'),
            ('bimodality', 'Bimodality Score', 'Secondary Criterion'),
            ('silhouette', 'Silhouette Score', 'Clustering Quality'),
            ('stability', 'Stability Score', 'Reproducibility'),
            ('reconstruction_error', 'Reconstruction Error', 'Fit Quality'),
            ('composite_score', 'Composite Score', 'Overall Score')
        ]
        
        for i, (metric, title, subtitle) in enumerate(metrics):
            ax = axes[i//3, i%3]
            ax.plot(k_values, results[metric], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('k (Number of Factors)')
            ax.set_ylabel(title)
            ax.set_title(f'{title}\n({subtitle})')
            ax.grid(True, alpha=0.3)
            
            # Highlight optimal k for composite score
            if metric == 'composite_score':
                optimal_k = self.select_optimal_k(results)
                optimal_score = results.loc[results['k']==optimal_k, metric].iloc[0]
                ax.scatter([optimal_k], [optimal_score], 
                          color='red', s=100, zorder=5, label=f'Optimal k={optimal_k}')
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    def analyze_optimal_factors(self, X, groups, optimal_k):
        """
        Detailed analysis of factors at optimal k.
        """
        print(f"\nAnalyzing NMF factors for optimal k={optimal_k}")
        
        # Fit NMF with optimal k
        model = NMF(
            n_components=optimal_k,
            init='random',
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        W = model.fit_transform(X)
        H = model.components_
        
        # Create results DataFrames
        df_W = pd.DataFrame(W, columns=[f"Factor_{i+1}" for i in range(optimal_k)], index=X.index)
        df_H = pd.DataFrame(H, index=[f"Factor_{i+1}" for i in range(optimal_k)], columns=X.columns)
        
        # Add group information
        df_W['Group'] = groups
        
        # Analyze each factor
        factor_analysis = []
        for i in range(optimal_k):
            factor_name = f"Factor_{i+1}"
            factor_scores = W[:, i]
            
            # Correlation with groups
            group_encoded = pd.get_dummies(groups)
            max_corr = 0
            best_group = None
            
            for group_col in group_encoded.columns:
                group_vector = group_encoded[group_col].values
                corr, p_val = pearsonr(factor_scores, group_vector)
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_group = group_col
            
            # Bimodality assessment
            kurt = stats.kurtosis(factor_scores)
            bimodality = max(0, -kurt + 1)
            
            factor_analysis.append({
                'Factor': factor_name,
                'Best_Group_Correlation': max_corr,
                'Associated_Group': best_group,
                'Bimodality_Score': bimodality,
                'Mean_Usage': np.mean(factor_scores),
                'Std_Usage': np.std(factor_scores)
            })
        
        factor_df = pd.DataFrame(factor_analysis)
        
        # Plot factor analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Factor Analysis for Optimal k={optimal_k}', fontsize=16)
        
        # Factor usage heatmap
        ax1 = axes[0, 0]
        df_W_plot = df_W.set_index('Group')
        sns.heatmap(df_W_plot.T, cmap='viridis', ax=ax1, cbar_kws={'label': 'Usage Score'})
        ax1.set_title('Factor Usage by Sample Group')
        
        # Group correlations
        ax2 = axes[0, 1]
        factor_df.plot(x='Factor', y='Best_Group_Correlation', kind='bar', ax=ax2)
        ax2.set_title('Factor-Group Correlations')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.tick_params(axis='x', rotation=45)
        
        # Bimodality scores
        ax3 = axes[1, 0]
        factor_df.plot(x='Factor', y='Bimodality_Score', kind='bar', ax=ax3, color='orange')
        ax3.set_title('Factor Bimodality Scores')
        ax3.set_ylabel('Bimodality Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Usage distribution example
        ax4 = axes[1, 1]
        # Plot distribution of most discriminative factor
        best_factor_idx = factor_df['Best_Group_Correlation'].abs().idxmax()
        best_factor_name = factor_df.loc[best_factor_idx, 'Factor']
        
        for group in groups.unique():
            group_scores = df_W[df_W['Group'] == group][best_factor_name]
            ax4.hist(group_scores, alpha=0.7, label=group, bins=10)
        
        ax4.set_title(f'Usage Distribution: {best_factor_name}')
        ax4.set_xlabel('Usage Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        return df_W, df_H, factor_df, fig


def main():
    """
    Main analysis function - demonstrates the k-selection algorithm.
    """
    print("=== NMF K-Selection Analysis ===")
    
    # Initialize k-selector
    selector = NMFKSelector(random_state=42)
    
    # Load and preprocess data
    print("\n1. Loading data...")
    # Load counts with first column as gene names
    counts = pd.read_csv('data/counts.txt', sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata.txt', sep='\t', index_col=0)
    
    # Check if we need to align sample names between counts and metadata
    counts_samples = set(counts.columns)
    metadata_samples = set(metadata.index)
    
    # Find common samples
    common_samples = counts_samples.intersection(metadata_samples)
    print(f"Common samples between counts and metadata: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("No matching samples found. Using all available samples from metadata.")
        # Create simplified metadata for available samples
        available_samples = list(counts.columns)
        # Extract group information from sample names (simplified approach)
        groups_simple = []
        for sample in available_samples:
            if 'Mutant' in sample and 'Vehicle' in sample:
                groups_simple.append('Mutant_Vehicle')
            elif 'WT' in sample and 'Vehicle' in sample:
                groups_simple.append('WT_Vehicle')
            elif 'Mutant' in sample and 'DrugA' in sample:
                groups_simple.append('Mutant_DrugA')
            elif 'WT' in sample and 'DrugA' in sample:
                groups_simple.append('WT_DrugA')
            elif 'Mutant' in sample and 'DrugB' in sample:
                groups_simple.append('Mutant_DrugB')
            elif 'WT' in sample and 'DrugB' in sample:
                groups_simple.append('WT_DrugB')
            else:
                groups_simple.append('Unknown')
        
        metadata = pd.DataFrame({'Group': groups_simple}, index=available_samples)
    else:
        # Use only common samples
        counts = counts[list(common_samples)]
        metadata = metadata.loc[list(common_samples)]
    
    print(f"Raw data shape: {counts.shape}")
    print(f"Sample groups: {metadata['Group'].value_counts()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, groups = selector.preprocess_data(counts, metadata, n_variable_genes=3000)
    
    # Evaluate k range
    print("\n3. Evaluating k range...")
    k_range = range(2, 12)
    results = selector.evaluate_k_range(X, groups, k_range)
    
    # Select optimal k
    print("\n4. Selecting optimal k...")
    optimal_k = selector.select_optimal_k(results)
    
    # Plot results
    print("\n5. Generating visualizations...")
    fig1 = selector.plot_results(results)
    plt.savefig('nmf_k_selection_results.png', dpi=300, bbox_inches='tight')
    
    # Detailed analysis of optimal factors
    print("\n6. Analyzing optimal factors...")
    df_W, df_H, factor_analysis, fig2 = selector.analyze_optimal_factors(X, groups, optimal_k)
    plt.savefig('nmf_factor_analysis.png', dpi=300, bbox_inches='tight')
    
    # Save results
    print("\n7. Saving results...")
    results.to_csv('nmf_k_selection_results.csv', index=False)
    factor_analysis.to_csv('nmf_factor_analysis.csv', index=False)
    df_W.to_csv('nmf_sample_weights.csv')
    
    print("\nAnalysis complete!")
    print(f"Optimal k: {optimal_k}")
    print("Results saved to CSV files and PNG figures.")
    
    return selector, results, optimal_k, df_W, df_H

if __name__ == "__main__":
    selector, results, optimal_k, df_W, df_H = main()