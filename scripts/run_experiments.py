#!/usr/bin/env python3
"""
Main experiment runner for Persian LLM Evaluation

IMPORTANT LIMITATIONS:
1. Only MT task has genuine multi-variant (Persian/Dari/Tajik) coverage
2. Sentiment, NLI, and QA tasks evaluate Iranian Persian only
3. Results cannot generalize to cross-variant robustness outside MT
4. English prompts (if used) introduce confounding cross-lingual effects

DEFAULT BEHAVIOR: Evaluates on COMPLETE test sets for all tasks
Use --quick or --num-samples to limit sample size for testing

Usage:
    python scripts/run_experiments.py                    # Full evaluation (complete test sets)
    python scripts/run_experiments.py --quick            # Quick test (50 samples)
    python scripts/run_experiments.py --num-samples 500  # Custom sample limit
    python scripts/run_experiments.py --models "Qwen2.5-7B-Instruct,gemma-2-9b-it"
    python scripts/run_experiments.py --tasks "sentiment,nli"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed, create_directories, save_jsonl, logger
from src.data_loader import DataLoader, process_datasets
from src.models import ModelManager, run_inference
from src.evaluation import compute_all_metrics
from src.analysis import ResultsAnalyzer, create_paper_tables, create_paper_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Persian LLM Evaluation Experiments",
        epilog="""
Examples:
  # Full evaluation on complete test sets (RECOMMENDED for paper results)
  python scripts/run_experiments.py
  
  # Quick test with 50 samples per task
  python scripts/run_experiments.py --quick
  
  # Custom sample size
  python scripts/run_experiments.py --num-samples 500
  
  # Specific models and tasks
  python scripts/run_experiments.py --models "Qwen2.5-3B-Instruct,bloomz-1b7" --tasks "sentiment,mt"
  
  # Full deterministic evaluation
  python scripts/run_experiments.py --deterministic --seed 42

Note: By default, evaluation uses COMPLETE test sets. Use --quick or --num-samples 
for faster testing during development.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to evaluate (default: all)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks (default: all enabled)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples per task (overrides config)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with 50 samples per task"
    )
    parser.add_argument(
        "--process-data-only",
        action="store_true",
        help="Only process raw data, skip inference"
    )
    parser.add_argument(
        "--skip-data-processing",
        action="store_true",
        help="Skip data processing (use existing processed data)"
    )
    parser.add_argument(
        "--only-analysis",
        action="store_true",
        help="Only run analysis on existing results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable full deterministic mode (may be slower)"
    )
    
    return parser.parse_args()


def estimate_runtime(num_models: int, num_tasks: int, num_samples: Optional[int]) -> str:
    """
    Estimate total runtime based on parameters.
    
    Args:
        num_samples: Number of samples per task, or None for full dataset
    """
    if num_samples is None:
        return "Variable (depends on dataset sizes - expect several hours for full evaluation)"
    
    # Rough estimates: ~2 seconds per sample per model
    total_samples = num_models * num_tasks * num_samples
    estimated_seconds = total_samples * 2
    
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    
    return f"{hours}h {minutes}m"


def run_task_evaluation(
    model_manager: ModelManager,
    data_loader: DataLoader,
    model_name: str,
    task: str,
    config: Dict,
    num_samples: Optional[int],
    predictions_dir: Path,
    metrics_dir: Path
) -> Dict:
    """Run evaluation for a single model on a single task."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name} on {task}")
    
    # Log variant availability
    task_config = config['tasks'][task]
    variants_available = task_config.get('variants_available', ['unknown'])
    logger.info(f"Task variants available: {variants_available}")
    
    if len(variants_available) == 1 and task != 'mt':
        logger.warning(
            f"⚠️  Task '{task}' has single-variant coverage only. "
            f"Cross-variant analysis will not be meaningful."
        )
    
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Load model
    model = model_manager.get_model(model_name)
    
    # Determine variants to evaluate
    if task == 'mt' and len(variants_available) > 1:
        variants = variants_available
    else:
        variants = [None]  # No variant filtering for single-variant tasks
    
    all_results = []
    
    for variant in variants:
        if variant:
            logger.info(f"\nProcessing variant: {variant}")
        
        # Calculate samples per variant
        if num_samples is not None:
            samples_per_variant = num_samples // len(variants) if len(variants) > 1 else num_samples
        else:
            samples_per_variant = None  # Use all available data
        
        # Load data
        data = data_loader.load_task_data(
            task=task,
            split='test',
            variant=variant,
            num_samples=samples_per_variant
        )
        
        if not data:
            logger.warning(f"No data found for {task}" + (f"/{variant}" if variant else ""))
            continue
        
        if samples_per_variant is None:
            logger.info(f"Loaded COMPLETE TEST SET: {len(data)} samples")
        else:
            logger.info(f"Loaded {len(data)} samples (limited)")
        
        # Log source distribution
        sources = {}
        for d in data:
            src = d.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        logger.info(f"Source distribution: {sources}")
        
        # Run inference
        results = run_inference(model, task, data, config)
        
        # Tag results with variant
        for r in results:
            if variant:
                r['variant'] = variant
        
        all_results.extend(results)
    
    # Save predictions
    predictions_path = predictions_dir / f"{model_name}_{task}_predictions.jsonl"
    save_jsonl(all_results, str(predictions_path))
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Compute metrics
    metrics = compute_all_metrics(all_results, task, config, model_name)
    
    # Add timing info
    elapsed = time.time() - start_time
    metrics['inference_time_seconds'] = elapsed
    
    # Log parsing issues if any
    if 'parsing_failure_rate' in metrics:
        failure_rate = metrics['parsing_failure_rate']
        if failure_rate > 0.1:
            logger.warning(
                f"⚠️  High parsing failure rate: {failure_rate*100:.1f}%. "
                f"Model outputs may be too verbose or ambiguous."
            )
    
    # Save metrics
    metrics_path = metrics_dir / f"{model_name}_{task}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Task completed in {elapsed/60:.1f} minutes")
    
    # Unload model to free memory
    model_manager.unload_model(model_name)
    
    return metrics


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override settings from command line
    if args.quick:
        config['evaluation']['num_samples'] = 50
        config['evaluation']['use_full_dataset'] = False
    elif args.num_samples:
        config['evaluation']['num_samples'] = args.num_samples
        config['evaluation']['use_full_dataset'] = False
    # If neither quick nor num_samples specified, use full dataset
    elif config['evaluation'].get('use_full_dataset', True):
        config['evaluation']['num_samples'] = None
        
    if args.seed:
        config['evaluation']['seed'] = args.seed
    if args.deterministic:
        config['evaluation']['deterministic'] = True
    
    # Set seed with deterministic mode if requested
    set_seed(config['evaluation']['seed'])
    
    # Create directories
    create_directories(config)
    
    # Setup paths
    predictions_dir = Path(config['paths']['predictions_dir'])
    metrics_dir = Path(config['paths']['metrics_dir'])
    
    # Determine models and tasks to run
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    else:
        models = [m['name'] for m in config['models']]
    
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        tasks = [t for t, tc in config['tasks'].items() if tc.get('enabled', True)]
    
    num_samples = config['evaluation'].get('num_samples')
    use_full_dataset = config['evaluation'].get('use_full_dataset', True)
    
    # Print configuration with warnings
    logger.info("\n" + "="*60)
    logger.info("PERSIAN LLM EVALUATION")
    logger.info("="*60)
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {tasks}")
    
    if use_full_dataset and num_samples is None:
        logger.info(f"Sample size: FULL DATASET (all available test samples)")
    else:
        logger.info(f"Sample size: {num_samples} per task (LIMITED)")
        
    logger.info(f"Seed: {config['evaluation']['seed']}")
    logger.info(f"Deterministic mode: {config['evaluation'].get('deterministic', False)}")
    
    if not use_full_dataset or num_samples is not None:
        logger.info(f"Estimated runtime: {estimate_runtime(len(models), len(tasks), num_samples or 300)}")
    else:
        logger.info("Runtime: Variable (depends on dataset sizes)")
    
    # Print scope warnings
    logger.info("\n" + "="*60)
    logger.info("SCOPE LIMITATIONS")
    logger.info("="*60)
    
    for task in tasks:
        task_config = config['tasks'][task]
        variants = task_config.get('variants_available', ['unknown'])
        prompt_lang = task_config.get('prompt_language', 'unknown')
        
        logger.info(f"\n{task.upper()}:")
        logger.info(f"  - Variants available: {variants}")
        logger.info(f"  - Prompt language: {prompt_lang}")
        
        if len(variants) == 1:
            logger.warning(
                f"  ⚠️  SINGLE VARIANT ONLY. Cross-variant robustness cannot be assessed."
            )
        
        if prompt_lang == 'english':
            logger.warning(
                f"  ⚠️  English prompts introduce cross-lingual confounds."
            )
    
    logger.info("\n" + "="*60 + "\n")
    
    # Only analysis mode
    if args.only_analysis:
        logger.info("Running analysis only...")
        create_paper_tables(config, config['paths']['results_dir'])
        create_paper_figures(config, config['paths']['results_dir'])
        logger.info("Analysis complete!")
        return
    
    # Initialize data loader and process raw data
    data_loader = DataLoader(config)
    
    if not args.skip_data_processing:
        logger.info("\n" + "="*60)
        logger.info("PROCESSING RAW DATA")
        logger.info("="*60)
        data_loader.process_all_datasets()
        
        # Print data statistics
        stats = data_loader.get_data_stats()
        logger.info("\nData Statistics (Complete Datasets):")
        for task, task_stats in stats.items():
            if task_stats['processed']:
                logger.info(f"\n  {task}: {task_stats['total']} samples")
                logger.info(f"    Splits: {task_stats['by_split']}")
                logger.info(f"    Variants: {task_stats['by_variant']}")
                logger.info(f"    Sources: {task_stats['by_source']}")
                
                # Show test set size specifically
                test_size = task_stats['by_split'].get('test', 0)
                if test_size > 0:
                    if use_full_dataset and num_samples is None:
                        logger.info(f"    ➜ Will evaluate on FULL TEST SET: {test_size} samples")
                    elif num_samples is not None:
                        logger.info(f"    ➜ Will evaluate on LIMITED: {min(num_samples, test_size)} samples")
        
        logger.info("\n" + "="*60)
    
    if args.process_data_only:
        logger.info("Data processing complete. Exiting.")
        return
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    # Run experiments
    all_metrics = {}
    total_start = time.time()
    
    for model_name in models:
        for task in tasks:
            try:
                metrics = run_task_evaluation(
                    model_manager=model_manager,
                    data_loader=data_loader,
                    model_name=model_name,
                    task=task,
                    config=config,
                    num_samples=num_samples,
                    predictions_dir=predictions_dir,
                    metrics_dir=metrics_dir
                )
                all_metrics[f"{model_name}_{task}"] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {task}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_elapsed = time.time() - total_start
    
    # Generate analysis and figures
    logger.info("\n" + "="*60)
    logger.info("GENERATING ANALYSIS AND FIGURES")
    logger.info("="*60)
    
    try:
        tables = create_paper_tables(config, config['paths']['results_dir'])
        
        # Print summary tables with interpretation notes
        for task in tasks:
            if f'{task}_main' in tables and not tables[f'{task}_main'].empty:
                logger.info(f"\n{task.upper()} Results:")
                print(tables[f'{task}_main'].to_string())
                
                # Add interpretation notes
                task_variants = config['tasks'][task].get('variants_available', [])
                if len(task_variants) == 1:
                    logger.warning(
                        f"\n⚠️  Note: {task.upper()} results are for {task_variants[0]} only. "
                        f"Do not generalize to other Persian variants."
                    )
            
            # Print source breakdown if available
            if f'{task}_sources' in tables and not tables[f'{task}_sources'].empty:
                logger.info(f"\n{task.upper()} Results by Dataset Source:")
                print(tables[f'{task}_sources'].to_string())
            
            # Print variant breakdown if available  
            if f'{task}_variants' in tables and not tables[f'{task}_variants'].empty:
                logger.info(f"\n{task.upper()} Results by Language Variant:")
                print(tables[f'{task}_variants'].to_string())
        
        create_paper_figures(config, config['paths']['results_dir'])
        
    except Exception as e:
        logger.warning(f"Could not generate some analysis outputs: {e}")
        import traceback
        traceback.print_exc()
    
    # Save experiment summary with detailed metadata
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runtime_minutes': total_elapsed / 60,
        'config': {
            'models': models,
            'tasks': tasks,
            'num_samples': num_samples,
            'use_full_dataset': use_full_dataset,
            'seed': config['evaluation']['seed'],
            'deterministic': config['evaluation'].get('deterministic', False),
            'normalization_applied': config.get('normalization', {}).get('apply_normalization', False)
        },
        'scope_limitations': {
            task: {
                'variants_available': config['tasks'][task].get('variants_available', []),
                'prompt_language': config['tasks'][task].get('prompt_language', 'unknown'),
                'multi_variant_coverage': len(config['tasks'][task].get('variants_available', [])) > 1
            }
            for task in tasks
        },
        'dataset_coverage': {
            'using_full_test_sets': use_full_dataset and num_samples is None,
            'sample_limitation': num_samples if num_samples is not None else 'none'
        },
        'results_summary': {
            k: {
                'accuracy': v.get('accuracy'),
                'effective_accuracy': v.get('effective_accuracy'),
                'parsing_failure_rate': v.get('parsing_failure_rate'),
                'bleu': v.get('bleu'),
                'f1': v.get('f1'),
                'exact_match': v.get('exact_match'),
                'inference_time': v.get('inference_time_seconds'),
                'source_analysis': v.get('source_analysis'),
                'variant_breakdown': v.get('variant_breakdown'),
                'notes': [
                    note for note in [
                        v.get('note'),
                        v.get('variant_note'),
                        v.get('cross_lingual_note'),
                        v.get('domain_note')
                    ] if note
                ]
            }
            for k, v in all_metrics.items()
        }
    }
    
    summary_path = Path(config['paths']['results_dir']) / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    logger.info(f"Results saved to: {config['paths']['results_dir']}")
    logger.info(f"Summary: {summary_path}")
    
    # Quick results overview with parsing stats
    logger.info("\nResults Overview:")
    for key, metrics in all_metrics.items():
        model, task = key.rsplit('_', 1)
        num_evaluated = metrics.get('num_samples', 0)
        
        if task in ['sentiment', 'nli']:
            score = metrics.get('accuracy', 0) * 100
            eff_score = metrics.get('effective_accuracy', 0) * 100
            parse_fail = metrics.get('parsing_failure_rate', 0) * 100
            logger.info(
                f"  {model}/{task} (n={num_evaluated}): {score:.1f}% accuracy "
                f"(effective: {eff_score:.1f}%, parsing failures: {parse_fail:.1f}%)"
            )
        elif task == 'mt':
            bleu = metrics.get('bleu', 0)
            logger.info(f"  {model}/{task} (n={num_evaluated}): {bleu:.1f} BLEU")
            # Print variant breakdown for MT
            if 'variant_breakdown' in metrics:
                for var, var_metrics in metrics['variant_breakdown'].items():
                    var_bleu = var_metrics.get('bleu', 0) or 0
                    var_n = var_metrics.get('num_samples', 0)
                    logger.info(f"    - {var} (n={var_n}): {var_bleu:.1f} BLEU")
        elif task == 'qa':
            f1 = metrics.get('f1', 0)
            logger.info(f"  {model}/{task} (n={num_evaluated}): {f1:.1f} F1")
    
    # Print final warnings
    logger.info("\n" + "="*60)
    logger.info("INTERPRETATION GUIDANCE")
    logger.info("="*60)
    
    if use_full_dataset and num_samples is None:
        logger.info("✓ Evaluation used COMPLETE TEST SETS for all tasks")
    else:
        logger.warning(
            f"⚠️  Evaluation used LIMITED samples ({num_samples} per task). "
            f"Results may not be representative of full dataset performance."
        )
    
    logger.info(
        "\nKey limitations:\n"
        "1. Only MT task has genuine multi-variant coverage\n"
        "2. Sentiment/NLI/QA results are for Iranian Persian only\n"
        "3. High parsing failure rates indicate output formatting issues\n"
        "4. Cross-lingual prompts may confound language understanding assessment\n"
        "5. Results aggregate across different source datasets/domains"
    )
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()