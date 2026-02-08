"""
Analysis and visualization for Persian LLM Evaluation
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150
})


class ResultsAnalyzer:
    """
    Analyze and visualize experimental results.
    """
    
    def __init__(self, config: Dict, results_dir: str = "results"):
        self.config = config
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.figures_dir = self.results_dir / "figures"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_results(self) -> Dict[str, Dict]:
        """Load all metric files."""
        results = {}
        
        for metric_file in self.metrics_dir.glob("*.json"):
            with open(metric_file, 'r') as f:
                data = json.load(f)
                key = metric_file.stem  # e.g., "aya-23-8B_sentiment"
                results[key] = data
        
        return results
    
    def create_summary_table(
        self,
        results: Dict[str, Dict],
        task: str
    ) -> pd.DataFrame:
        """
        Create summary table for a task across all models.
        """
        rows = []
        
        for key, metrics in results.items():
            if task not in key:
                continue
            
            model_name = key.replace(f"_{task}", "")
            
            row = {
                'Model': model_name,
            }
            
            if task in ['sentiment', 'nli']:
                row['Accuracy'] = metrics.get('accuracy', 0) * 100
                row['F1'] = metrics.get('f1', 0) * 100
                row['Unknown Rate'] = metrics.get('unknown_rate', 0) * 100
            elif task == 'mt':
                row['BLEU'] = metrics.get('bleu', 0)
                row['chrF'] = metrics.get('chrf', 0)
            elif task == 'qa':
                row['EM'] = metrics.get('exact_match', 0)
                row['F1'] = metrics.get('f1', 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Model')
        
        return df
    
    def create_variant_comparison_table(
        self,
        results: Dict[str, Dict],
        task: str
    ) -> pd.DataFrame:
        """
        Create table comparing performance across variants.
        """
        rows = []
        
        for key, metrics in results.items():
            if task not in key:
                continue
            if 'variant_analysis' not in metrics:
                continue
            
            model_name = key.replace(f"_{task}", "")
            variant_analysis = metrics['variant_analysis']
            
            row = {
                'Model': model_name,
                'Mean Score': variant_analysis.get('mean_score', 0),
                'Std': variant_analysis.get('std_score', 0),
                'Performance Gap': variant_analysis.get('performance_gap', 0),
                'Best Variant': variant_analysis.get('best_variant', 'N/A'),
                'Worst Variant': variant_analysis.get('worst_variant', 'N/A'),
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict],
        task: str,
        metric: str,
        save_path: Optional[str] = None
    ):
        """
        Create bar plot comparing models on a metric.
        """
        df = self.create_summary_table(results, task)
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data for {metric} in {task}")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = sns.color_palette("husl", len(df))
        bars = ax.bar(df['Model'], df[metric], color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{task.upper()} Task: {metric} by Model')
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / f'{task}_{metric}_comparison.pdf', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: List[List[int]],
        labels: List[str],
        model_name: str,
        task: str,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap.
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        
        cm_array = np.array(confusion_matrix)
        
        # Normalize
        cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model_name} - {task.upper()} Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(
                self.figures_dir / f'{model_name}_{task}_confusion.pdf',
                dpi=300, bbox_inches='tight'
            )
        
        plt.close()
    
    def plot_variant_performance(
        self,
        results: Dict[str, Dict],
        task: str,
        save_path: Optional[str] = None
    ):
        """
        Plot performance across variants for all models.
        """
        # Collect variant data
        variant_data = []
        
        for key, metrics in results.items():
            if task not in key:
                continue
            if 'variant_analysis' not in metrics:
                continue
            
            model_name = key.replace(f"_{task}", "")
            variant_scores = metrics['variant_analysis'].get('variant_scores', {})
            
            for variant, score in variant_scores.items():
                variant_data.append({
                    'Model': model_name,
                    'Variant': variant,
                    'Score': score * 100 if score <= 1 else score
                })
        
        if not variant_data:
            logger.warning(f"No variant data for {task}")
            return
        
        df = pd.DataFrame(variant_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar plot
        variants = df['Variant'].unique()
        models = df['Model'].unique()
        x = np.arange(len(models))
        width = 0.8 / len(variants)
        
        colors = sns.color_palette("Set2", len(variants))
        
        for i, variant in enumerate(variants):
            variant_df = df[df['Variant'] == variant]
            scores = [
                variant_df[variant_df['Model'] == m]['Score'].values[0]
                if m in variant_df['Model'].values else 0
                for m in models
            ]
            ax.bar(x + i * width, scores, width, label=variant, color=colors[i])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score (%)')
        ax.set_title(f'{task.upper()} Performance Across Variants')
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(
                self.figures_dir / f'{task}_variant_comparison.pdf',
                dpi=300, bbox_inches='tight'
            )
        
        plt.close()
    
    def plot_error_analysis(
        self,
        results: Dict[str, Dict],
        task: str,
        save_path: Optional[str] = None
    ):
        """
        Plot error type distribution across models.
        """
        error_data = []
        
        for key, metrics in results.items():
            if task not in key:
                continue
            
            model_name = key.replace(f"_{task}", "")
            error_analysis = metrics.get('error_analysis', {})
            top_errors = error_analysis.get('top_errors', [])
            
            for error_type, count in top_errors[:3]:  # Top 3 errors
                error_data.append({
                    'Model': model_name,
                    'Error Type': error_type,
                    'Count': count
                })
        
        if not error_data:
            logger.warning(f"No error data for {task}")
            return
        
        df = pd.DataFrame(error_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create stacked bar chart
        pivot_df = df.pivot_table(
            values='Count',
            index='Model',
            columns='Error Type',
            fill_value=0
        )
        
        pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Count')
        ax.set_title(f'{task.upper()} Error Distribution by Model')
        ax.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(
                self.figures_dir / f'{task}_error_analysis.pdf',
                dpi=300, bbox_inches='tight'
            )
        
        plt.close()
    
    def generate_latex_table(
        self,
        results: Dict[str, Dict],
        task: str
    ) -> str:
        """
        Generate LaTeX table for the paper.
        """
        df = self.create_summary_table(results, task)
        
        if df.empty:
            return "% No data available"
        
        # Format for LaTeX
        latex = df.to_latex(
            index=False,
            float_format="%.2f",
            caption=f"Performance comparison on {task.upper()} task",
            label=f"tab:{task}_results"
        )
        
        return latex
    
    def generate_all_figures(self, results: Dict[str, Dict]):
        """Generate all figures for the paper."""
        
        tasks = ['sentiment', 'mt', 'nli', 'qa']
        
        for task in tasks:
            task_results = {k: v for k, v in results.items() if task in k}
            
            if not task_results:
                continue
            
            logger.info(f"Generating figures for {task}")
            
            # Main metric comparison
            if task in ['sentiment', 'nli']:
                self.plot_model_comparison(results, task, 'Accuracy')
            elif task == 'mt':
                self.plot_model_comparison(results, task, 'BLEU')
            elif task == 'qa':
                self.plot_model_comparison(results, task, 'F1')
            
            # Variant comparison
            self.plot_variant_performance(results, task)
            
            # Error analysis (for classification tasks)
            if task in ['sentiment', 'nli']:
                self.plot_error_analysis(results, task)
                
                # Confusion matrices
                for key, metrics in task_results.items():
                    if 'confusion_matrix' in metrics:
                        model_name = key.replace(f"_{task}", "")
                        labels = self.config['tasks'][task]['labels']
                        self.plot_confusion_matrix(
                            metrics['confusion_matrix'],
                            labels,
                            model_name,
                            task
                        )
        
        logger.info("All figures generated!")


def create_paper_tables(config: Dict, results_dir: str = "results"):
    """
    Create all tables needed for the paper.
    """
    analyzer = ResultsAnalyzer(config, results_dir)
    results = analyzer.load_all_results()
    
    tables = {}
    
    for task in ['sentiment', 'mt', 'nli', 'qa']:
        # Main results table
        tables[f'{task}_main'] = analyzer.create_summary_table(results, task)
        
        # Variant comparison table
        tables[f'{task}_variants'] = analyzer.create_variant_comparison_table(results, task)
        
        # LaTeX version
        tables[f'{task}_latex'] = analyzer.generate_latex_table(results, task)
    
    return tables


def create_paper_figures(config: Dict, results_dir: str = "results"):
    """
    Create all figures needed for the paper.
    """
    analyzer = ResultsAnalyzer(config, results_dir)
    results = analyzer.load_all_results()
    analyzer.generate_all_figures(results)