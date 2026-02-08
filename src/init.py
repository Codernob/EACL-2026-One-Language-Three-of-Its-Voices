"""
Persian LLM Evaluation Package
"""

from .utils import load_config, set_seed, create_directories
from .data_loader import DataLoader, process_datasets
from .models import MultilingualLLM, ModelManager
from .evaluation import Evaluator, RobustnessAnalyzer, compute_all_metrics
from .analysis import ResultsAnalyzer, create_paper_tables, create_paper_figures

__version__ = "1.0.0"
__author__ = "codernob"

__all__ = [
    'load_config',
    'set_seed', 
    'create_directories',
    'DataLoader',
    'process_datasets',
    'MultilingualLLM',
    'ModelManager',
    'Evaluator',
    'RobustnessAnalyzer',
    'compute_all_metrics',
    'ResultsAnalyzer',
    'create_paper_tables',
    'create_paper_figures',
]