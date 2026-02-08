"""
Evaluation metrics for Persian LLM Evaluation
"""

import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import sacrebleu
import logging

from src.utils import extract_label, normalize_persian_text

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Unified evaluator for all tasks.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def evaluate(
        self,
        task: str,
        results: List[Dict],
        model_name: str
    ) -> Dict[str, any]:
        """
        Evaluate results for a specific task.
        
        Args:
            task: Task name
            results: List of result dictionaries
            model_name: Name of the model being evaluated
        
        Returns:
            Dictionary of metrics
        """
        if task == 'sentiment':
            return self._evaluate_classification(
                results,
                labels=self.config['tasks']['sentiment']['labels'],
                task='sentiment'
            )
        elif task == 'mt':
            return self._evaluate_mt(results)
        elif task == 'nli':
            return self._evaluate_classification(
                results,
                labels=self.config['tasks']['nli']['labels'],
                task='nli'
            )
        elif task == 'qa':
            return self._evaluate_qa(results)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _evaluate_classification(
        self,
        results: List[Dict],
        labels: List[str],
        task: str
    ) -> Dict[str, any]:
        """Evaluate classification tasks (sentiment, NLI)."""
        
        predictions = []
        gold_labels = []
        
        for r in results:
            pred = extract_label(r['response'], labels, task=task)
            gold = r['gold']
            
            predictions.append(pred)
            gold_labels.append(gold)
        
        # Calculate metrics
        # Filter out unknown predictions for accuracy calculation
        valid_pairs = [(p, g) for p, g in zip(predictions, gold_labels) if p != 'unknown']
        
        if not valid_pairs:
            return {
                'accuracy': 0.0,
                'valid_predictions': 0,
                'total': len(results),
                'error_rate': 1.0
            }
        
        valid_preds, valid_golds = zip(*valid_pairs)
        
        accuracy = accuracy_score(valid_golds, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_golds, valid_preds, labels=labels, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(valid_golds, valid_preds, labels=labels)
        
        # Per-class metrics
        class_report = classification_report(
            valid_golds, valid_preds,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Error analysis
        errors = self._analyze_classification_errors(
            predictions, gold_labels, labels
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'per_class': class_report,
            'valid_predictions': len(valid_pairs),
            'total': len(results),
            'unknown_rate': (len(results) - len(valid_pairs)) / len(results),
            'error_analysis': errors
        }
    
    def _analyze_classification_errors(
        self,
        predictions: List[str],
        gold_labels: List[str],
        labels: List[str]
    ) -> Dict[str, any]:
        """Analyze classification errors for patterns."""
        
        error_types = {f"{g}->{p}": 0 for g in labels for p in labels if g != p}
        error_types['unknown'] = 0
        
        for pred, gold in zip(predictions, gold_labels):
            if pred == 'unknown':
                error_types['unknown'] += 1
            elif pred != gold:
                key = f"{gold}->{pred}"
                if key in error_types:
                    error_types[key] += 1
        
        # Most common errors
        sorted_errors = sorted(
            [(k, v) for k, v in error_types.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'error_distribution': error_types,
            'top_errors': sorted_errors[:5]
        }
    
    def _evaluate_mt(self, results: List[Dict]) -> Dict[str, any]:
        """Evaluate machine translation with BLEU and chrF."""
        
        hypotheses = []
        references = []
        
        for r in results:
            hyp = r['response'].strip()
            ref = r['gold'] if isinstance(r['gold'], str) else r['gold'][0]
            
            hypotheses.append(hyp)
            references.append([ref])  # sacrebleu expects list of references
        
        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))
        
        # Calculate chrF
        chrf = sacrebleu.corpus_chrf(hypotheses, list(zip(*references)))
        
        # Calculate per-sample BLEU for variance analysis
        sample_bleus = []
        for hyp, ref_list in zip(hypotheses, references):
            try:
                sample_bleu = sacrebleu.sentence_bleu(hyp, ref_list)
                sample_bleus.append(sample_bleu.score)
            except Exception:
                sample_bleus.append(0.0)
        
        return {
            'bleu': bleu.score,
            'chrf': chrf.score,
            'bleu_details': {
                'precisions': bleu.precisions,
                'brevity_penalty': bleu.bp,
                'length_ratio': bleu.sys_len / bleu.ref_len if bleu.ref_len > 0 else 0
            },
            'sample_bleu_mean': np.mean(sample_bleus),
            'sample_bleu_std': np.std(sample_bleus),
            'total': len(results)
        }
    
    def _evaluate_qa(self, results: List[Dict]) -> Dict[str, any]:
        """Evaluate question answering with EM and F1."""
        
        exact_matches = []
        f1_scores = []
        
        for r in results:
            prediction = r['response'].strip()
            gold_answers = r['gold']
            
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            # Calculate EM and F1 against all gold answers
            em = max(
                self._exact_match(prediction, gold)
                for gold in gold_answers
            )
            f1 = max(
                self._token_f1(prediction, gold)
                for gold in gold_answers
            )
            
            exact_matches.append(em)
            f1_scores.append(f1)
        
        return {
            'exact_match': np.mean(exact_matches) * 100,
            'f1': np.mean(f1_scores) * 100,
            'exact_match_std': np.std(exact_matches) * 100,
            'f1_std': np.std(f1_scores) * 100,
            'total': len(results)
        }
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        text = normalize_persian_text(text)
        text = text.lower()
        # Remove punctuation
        text = ''.join(c for c in text if c not in string.punctuation + '،؟')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _exact_match(self, prediction: str, gold: str) -> float:
        """Calculate exact match score."""
        return float(
            self._normalize_answer(prediction) == self._normalize_answer(gold)
        )
    
    def _token_f1(self, prediction: str, gold: str) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = self._normalize_answer(prediction).split()
        gold_tokens = self._normalize_answer(gold).split()
        
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1


class RobustnessAnalyzer:
    """
    Analyze model robustness across Persian variants and scripts.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def compare_variants(
        self,
        results_by_variant: Dict[str, Dict[str, any]],
        task: str
    ) -> Dict[str, any]:
        """
        Compare model performance across variants.
        
        Args:
            results_by_variant: {variant_name: metrics_dict}
            task: Task name
        
        Returns:
            Comparison analysis
        """
        if task in ['sentiment', 'nli']:
            metric = 'accuracy'
        elif task == 'mt':
            metric = 'bleu'
        elif task == 'qa':
            metric = 'f1'
        else:
            metric = 'accuracy'
        
        # Extract metric values
        variant_scores = {
            variant: metrics.get(metric, 0)
            for variant, metrics in results_by_variant.items()
        }
        
        # Calculate statistics
        scores = list(variant_scores.values())
        
        analysis = {
            'variant_scores': variant_scores,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'score_range': max(scores) - min(scores) if scores else 0,
            'metric_used': metric
        }
        
        # Identify best and worst variants
        if variant_scores:
            analysis['best_variant'] = max(variant_scores, key=variant_scores.get)
            analysis['worst_variant'] = min(variant_scores, key=variant_scores.get)
            analysis['performance_gap'] = (
                variant_scores[analysis['best_variant']] -
                variant_scores[analysis['worst_variant']]
            )
        
        return analysis
    
    def analyze_script_sensitivity(
        self,
        standard_results: List[Dict],
        variant_results: List[Dict],
        task: str
    ) -> Dict[str, any]:
        """
        Analyze model sensitivity to script changes.
        """
        # Match results by ID (same content, different script)
        standard_by_id = {r['id'].replace('_tajik', ''): r for r in standard_results}
        
        consistency_count = 0
        total_pairs = 0
        
        for var_result in variant_results:
            base_id = var_result['id'].replace('_tajik', '')
            
            if base_id in standard_by_id:
                std_result = standard_by_id[base_id]
                total_pairs += 1
                
                # Compare predictions
                if task in ['sentiment', 'nli']:
                    std_pred = extract_label(
                        std_result['response'],
                        self.config['tasks'][task]['labels'],
                        task
                    )
                    var_pred = extract_label(
                        var_result['response'],
                        self.config['tasks'][task]['labels'],
                        task
                    )
                    if std_pred == var_pred:
                        consistency_count += 1
                else:
                    # For generation tasks, use approximate matching
                    if std_result['response'].strip()[:50] == var_result['response'].strip()[:50]:
                        consistency_count += 1
        
        consistency_rate = consistency_count / total_pairs if total_pairs > 0 else 0
        
        return {
            'consistency_rate': consistency_rate,
            'total_pairs': total_pairs,
            'consistent_pairs': consistency_count,
            'sensitivity': 1 - consistency_rate  # Higher = more sensitive to script
        }


def compute_all_metrics(
    results: List[Dict],
    task: str,
    config: Dict,
    model_name: str
) -> Dict[str, any]:
    """
    Compute all metrics for a set of results.
    Includes breakdown by source dataset and language variant.
    """
    evaluator = Evaluator(config)
    
    # Basic metrics
    metrics = evaluator.evaluate(task, results, model_name)
    
    # Add metadata
    metrics['model'] = model_name
    metrics['task'] = task
    metrics['num_samples'] = len(results)
    
    # Group by source dataset (e.g., sentipers, digikala, flores, tatoeba)
    sources = set(r.get('source', 'unknown') for r in results)
    if len(sources) > 1:
        metrics['source_analysis'] = {}
        for source in sources:
            source_results = [r for r in results if r.get('source') == source]
            if source_results:
                source_metrics = evaluator.evaluate(task, source_results, model_name)
                metrics['source_analysis'][source] = {
                    'num_samples': len(source_results),
                    'accuracy': source_metrics.get('accuracy'),
                    'f1': source_metrics.get('f1'),
                    'bleu': source_metrics.get('bleu'),
                    'exact_match': source_metrics.get('exact_match')
                }
    
    # Group by variant for robustness analysis (persian, dari, tajik)
    variants = set(r.get('variant', 'unknown') for r in results)
    
    if len(variants) > 1:
        results_by_variant = {}
        metrics['variant_breakdown'] = {}
        
        for variant in variants:
            variant_results = [r for r in results if r.get('variant') == variant]
            if variant_results:
                variant_metrics = evaluator.evaluate(task, variant_results, model_name)
                results_by_variant[variant] = variant_metrics
                
                # Store detailed breakdown
                metrics['variant_breakdown'][variant] = {
                    'num_samples': len(variant_results),
                    'accuracy': variant_metrics.get('accuracy'),
                    'f1': variant_metrics.get('f1'),
                    'bleu': variant_metrics.get('bleu'),
                    'chrf': variant_metrics.get('chrf'),
                    'exact_match': variant_metrics.get('exact_match')
                }
        
        analyzer = RobustnessAnalyzer(config)
        metrics['variant_analysis'] = analyzer.compare_variants(
            results_by_variant, task
        )
    
    return metrics