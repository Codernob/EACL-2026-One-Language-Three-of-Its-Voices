"""
Utility functions for Persian LLM Evaluation
"""

import os
import json
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directories(config: Dict):
    """Create necessary directories."""
    dirs = [
        config['paths']['data_dir'],
        config['paths']['raw_dir'],
        config['paths']['processed_dir'],
        config['paths']['results_dir'],
        config['paths']['predictions_dir'],
        config['paths']['metrics_dir'],
        config['paths']['figures_dir'],
    ]
    
    # Create subdirectories for raw data
    raw_subdirs = [
        'sentiment/persent',
        'sentiment/digikala', 
        'sentiment/snappfood',
        'mt/flores200',
        'mt/tatoeba',
        'nli/xnli',
        'qa/pquad'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    for subdir in raw_subdirs:
        Path(config['paths']['raw_dir']) / subdir
        (Path(config['paths']['raw_dir']) / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Created all directories")


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL format."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def normalize_persian_text(text: str) -> str:
    """
    Normalize Persian text for consistent evaluation.
    Handles common character variations.
    """
    # Common character normalizations
    replacements = {
        'ك': 'ک',  # Arabic kaf to Persian kaf
        'ي': 'ی',  # Arabic yeh to Persian yeh
        '٠': '۰', '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴',
        '٥': '۵', '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹',
        '\u200c': ' ',  # Zero-width non-joiner to space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_prompt(
    task: str,
    text: str,
    labels: Optional[List[str]] = None,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    context: Optional[str] = None,
    question: Optional[str] = None
) -> str:
    """
    Format prompts for different tasks.
    Returns zero-shot prompts for each task type.
    """
    
    if task == "sentiment":
        prompt = f"""Classify the sentiment of the following Persian text as one of: {', '.join(labels)}.

Text: {text}

Respond with only the sentiment label, nothing else.
Sentiment:"""
    
    elif task == "mt":
        lang_names = {
            "fas": "Persian",
            "prs": "Dari",
            "tgk": "Tajik",
            "eng": "English"
        }
        src = lang_names.get(source_lang, source_lang)
        tgt = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following {src} text to {tgt}.

{src} text: {text}

{tgt} translation:"""
    
    elif task == "nli":
        prompt = f"""Given the premise and hypothesis below, determine the relationship between them.
Choose one of: {', '.join(labels)}.

Premise: {text}
Hypothesis: {context}

Respond with only the label, nothing else.
Relationship:"""
    
    elif task == "qa":
        prompt = f"""Answer the question based on the given context. Extract the answer directly from the context.

Context: {context}

Question: {question}

Answer:"""
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return prompt


def extract_label(
    response: str,
    valid_labels: List[str],
    task: str = "classification"
) -> str:
    """
    Extract label from model response.
    Handles various response formats.
    """
    response = response.strip().lower()
    
    # Direct match
    for label in valid_labels:
        if label.lower() in response:
            return label
    
    # Partial match for sentiment
    if task == "sentiment":
        sentiment_mappings = {
            "positive": ["positive", "مثبت", "خوب", "pos"],
            "negative": ["negative", "منفی", "بد", "neg"],
            "neutral": ["neutral", "خنثی", "بیطرف", "neu"]
        }
        for label, variants in sentiment_mappings.items():
            if any(v in response for v in variants):
                return label
    
    # NLI mappings
    if task == "nli":
        nli_mappings = {
            "entailment": ["entailment", "entail", "yes", "true"],
            "contradiction": ["contradiction", "contradict", "no", "false"],
            "neutral": ["neutral", "maybe", "unknown"]
        }
        for label, variants in nli_mappings.items():
            if any(v in response for v in variants):
                return label
    
    # Return first word as fallback
    first_word = response.split()[0] if response.split() else ""
    for label in valid_labels:
        if first_word.startswith(label[:3].lower()):
            return label
    
    return "unknown"


class ScriptConverter:
    """
    Convert between Persian script variants.
    Useful for robustness testing.
    """
    
    # Mapping from Persian to Tajik Cyrillic (simplified)
    PERSIAN_TO_TAJIK = {
        'ا': 'а', 'آ': 'о', 'ب': 'б', 'پ': 'п', 'ت': 'т',
        'ث': 'с', 'ج': 'ҷ', 'چ': 'ч', 'ح': 'ҳ', 'خ': 'х',
        'د': 'д', 'ذ': 'з', 'ر': 'р', 'ز': 'з', 'ژ': 'ж',
        'س': 'с', 'ش': 'ш', 'ص': 'с', 'ض': 'з', 'ط': 'т',
        'ظ': 'з', 'ع': 'ъ', 'غ': 'ғ', 'ف': 'ф', 'ق': 'қ',
        'ک': 'к', 'گ': 'г', 'ل': 'л', 'م': 'м', 'ن': 'н',
        'و': 'в', 'ه': 'ҳ', 'ی': 'й', 'ے': 'е',
        ' ': ' ', '،': ',', '؟': '?', '!': '!'
    }
    
    @classmethod
    def persian_to_tajik_approx(cls, text: str) -> str:
        """
        Approximate conversion from Persian to Tajik script.
        Note: This is a simplified conversion for testing purposes.
        """
        result = []
        for char in text:
            result.append(cls.PERSIAN_TO_TAJIK.get(char, char))
        return ''.join(result)


def compute_text_statistics(texts: List[str]) -> Dict[str, float]:
    """Compute basic statistics about text data."""
    lengths = [len(t.split()) for t in texts]
    return {
        'num_samples': len(texts),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }