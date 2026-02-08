"""
Data loading and preprocessing for Persian LLM Evaluation
Handles: SentiPers, Digikala, SnappFood, Flores, Tatoeba, FarsTail, PQuAD
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.utils import save_jsonl, load_jsonl, normalize_persian_text

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for all Persian evaluation datasets.
    Processes raw data files into unified JSONL format.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_dir = Path(config['paths']['raw_dir'])
        self.processed_dir = Path(config['paths']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all_datasets(self):
        """Process all raw datasets into unified format."""
        logger.info("Processing all datasets from raw files...")
        
        self._process_sentiment_data()
        self._process_mt_data()
        self._process_nli_data()
        self._process_qa_data()
        
        logger.info("All datasets processed successfully!")
    
    # ==================== SENTIMENT ====================
    
    def _process_sentiment_data(self):
        """Process all sentiment datasets."""
        logger.info("Processing sentiment datasets...")
        
        all_sentiment_data = []
        
        # 1. SentiPers
        sentipers_data = self._load_sentipers()
        all_sentiment_data.extend(sentipers_data)
        
        # 2. Digikala
        digikala_data = self._load_digikala()
        all_sentiment_data.extend(digikala_data)
        
        # 3. SnappFood
        snappfood_data = self._load_snappfood()
        all_sentiment_data.extend(snappfood_data)
        
        # Save combined sentiment data
        if all_sentiment_data:
            save_jsonl(all_sentiment_data, str(self.processed_dir / 'sentiment.jsonl'))
            logger.info(f"Saved {len(all_sentiment_data)} total sentiment samples")
    
    def _load_sentipers(self) -> List[Dict]:
        """
        Load SentiPers dataset from sentipers.xlsx
        Polarity: -2 (very negative) to +2 (very positive), 0 = neutral
        """
        filepath = self.raw_dir / 'sentiment' / 'sentipers' / 'sentipers.xlsx'
        
        if not filepath.exists():
            logger.warning(f"SentiPers not found at {filepath}")
            return []
        
        try:
            df = pd.read_excel(filepath)
            logger.info(f"Loaded SentiPers with {len(df)} rows")
            
            data = []
            for idx, row in df.iterrows():
                text = str(row['text']).strip()
                polarity = row['polarity']
                
                # Map polarity to 3-class labels
                if polarity in [-2, -1, '-2', '-1']:
                    label = 'negative'
                elif polarity in [0, '0']:
                    label = 'neutral'
                elif polarity in [1, 2, '+1', '+2', '1', '2']:
                    label = 'positive'
                else:
                    continue
                
                data.append({
                    'id': f"sentipers_{idx}",
                    'text': text,  # Don't normalize yet - preserve variants
                    'label': label,
                    'original_label': str(polarity),
                    'source': 'sentipers',
                    'split': 'test' if idx % 10 == 0 else 'train',
                    'variant': 'persian_standard'
                })
            
            logger.info(f"Processed {len(data)} SentiPers samples")
            return data
            
        except Exception as e:
            logger.error(f"Error loading SentiPers: {e}")
            return []
    
    def _load_digikala(self) -> List[Dict]:
        """Load Digikala dataset"""
        filepath = self.raw_dir / 'sentiment' / 'digikala' / 'data.xls'
        
        if not filepath.exists():
            logger.warning(f"Digikala not found at {filepath}")
            return []
        
        try:
            xls = pd.ExcelFile(filepath)
            df1 = pd.read_excel(xls, 'Export')
            df2 = pd.read_excel(xls, 'Sheet 2')
            df = pd.concat([df1, df2]).reset_index(drop=True)
            
            if 'verification_status' in df.columns:
                df = df[df['verification_status'] == 'verified']
            
            logger.info(f"Loaded Digikala with {len(df)} rows")
            
            label_map = {0: 'negative', 0.0: 'negative',
                        1: 'neutral', 1.0: 'neutral',
                        2: 'positive', 2.0: 'positive'}
            
            data = []
            for idx, row in df.iterrows():
                text = str(row.get('comment', '')).strip()
                if not text or text == 'nan':
                    continue
                
                label_val = row.get('label')
                if label_val not in label_map:
                    continue
                
                label = label_map[label_val]
                
                data.append({
                    'id': f"digikala_{idx}",
                    'text': text,  # Don't normalize yet - preserve variants
                    'label': label,
                    'source': 'digikala',
                    'split': 'test' if idx % 10 == 0 else 'train',
                    'variant': 'persian_standard'
                })
            
            logger.info(f"Processed {len(data)} Digikala samples")
            return data
            
        except Exception as e:
            logger.error(f"Error loading Digikala: {e}")
            return []
    
    def _load_snappfood(self) -> List[Dict]:
        """Load SnappFood dataset"""
        snappfood_dir = self.raw_dir / 'sentiment' / 'snappfood'
        
        if not snappfood_dir.exists():
            logger.warning(f"SnappFood directory not found at {snappfood_dir}")
            return []
        
        data = []
        
        label_map = {
            'HAPPY': 'positive', 'happy': 'positive',
            'SAD': 'negative', 'sad': 'negative',
            0: 'positive', 1: 'negative'
        }
        
        for split_name, filename in [('train', 'train.csv'), ('test', 'test.csv'), ('dev', 'dev.csv')]:
            filepath = snappfood_dir / filename
            
            if not filepath.exists():
                logger.warning(f"SnappFood {filename} not found")
                continue
            
            try:
                df = None
                for sep in ['\t', ',', ';']:
                    try:
                        df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip')
                        if len(df.columns) > 1:
                            break
                    except:
                        continue
                
                if df is None or len(df.columns) <= 1:
                    logger.warning(f"SnappFood {filename}: Could not parse")
                    continue
                
                logger.info(f"Loaded SnappFood {filename} with {len(df)} rows")
                
                text_col = None
                label_col = None
                col_lower_map = {col.lower().strip(): col for col in df.columns}
                
                for key in ['comment', 'text', 'review']:
                    if key in col_lower_map:
                        text_col = col_lower_map[key]
                        break
                
                for key in ['label', 'label_id', 'sentiment']:
                    if key in col_lower_map:
                        label_col = col_lower_map[key]
                        break
                
                if text_col is None or label_col is None:
                    logger.warning(f"SnappFood {filename}: Missing columns")
                    continue
                
                for idx, row in df.iterrows():
                    text = str(row.get(text_col, '')).strip()
                    if not text or text == 'nan':
                        continue
                    
                    label_val = row.get(label_col)
                    if label_val not in label_map:
                        continue
                    
                    label = label_map[label_val]
                    
                    data.append({
                        'id': f"snappfood_{split_name}_{idx}",
                        'text': text,  # Don't normalize yet - preserve variants
                        'label': label,
                        'source': 'snappfood',
                        'split': 'test' if split_name == 'test' else split_name,
                        'variant': 'persian_standard'
                    })
                    
            except Exception as e:
                logger.error(f"Error loading SnappFood {filename}: {e}")
        
        logger.info(f"Processed {len(data)} SnappFood samples")
        return data
    
    # ==================== MACHINE TRANSLATION ====================
    
    def _process_mt_data(self):
        """Process all MT datasets."""
        logger.info("Processing MT datasets...")
        
        all_mt_data = []
        
        flores_data = self._load_flores()
        all_mt_data.extend(flores_data)
        
        tatoeba_data = self._load_tatoeba()
        all_mt_data.extend(tatoeba_data)
        
        if all_mt_data:
            save_jsonl(all_mt_data, str(self.processed_dir / 'mt.jsonl'))
            logger.info(f"Saved {len(all_mt_data)} total MT samples")
    
    def _load_flores(self) -> List[Dict]:
        """Load Flores-200 dataset"""
        flores_dir = self.raw_dir / 'mt' / 'flores200'
        
        if not flores_dir.exists():
            logger.warning(f"Flores directory not found at {flores_dir}")
            return []
        
        eng_file = flores_dir / 'eng_Latn.devtest'
        if not eng_file.exists():
            logger.warning("English Flores file not found")
            return []
        
        with open(eng_file, 'r', encoding='utf-8') as f:
            eng_sentences = [line.strip() for line in f.readlines()]
        
        data = []
        
        source_files = {
            'persian': ('pes_Arab.devtest', 'pes'),
            'dari': ('prs_Arab.devtest', 'prs'),
            'tajik': ('tgk_Cyrl.devtest', 'tgk')
        }
        
        for variant, (filename, lang_code) in source_files.items():
            filepath = flores_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Flores {filename} not found")
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                source_sentences = [line.strip() for line in f.readlines()]
            
            for idx, (src, tgt) in enumerate(zip(source_sentences, eng_sentences)):
                if src and tgt:
                    data.append({
                        'id': f"flores_{variant}_{idx}",
                        'source_text': src,
                        'target_text': tgt,
                        'source_lang': lang_code,
                        'target_lang': 'eng',
                        'source': 'flores200',
                        'split': 'test',
                        'variant': variant
                    })
            
            logger.info(f"Loaded {len(source_sentences)} Flores {variant} samples")
        
        logger.info(f"Processed {len(data)} Flores MT samples")
        return data
    
    def _load_tatoeba(self) -> List[Dict]:
        """Load Tatoeba dataset"""
        tatoeba_dir = self.raw_dir / 'mt' / 'tatoeba'
        
        en_file = tatoeba_dir / 'en.txt'
        pes_file = tatoeba_dir / 'pes.txt'
        
        if not en_file.exists() or not pes_file.exists():
            logger.warning(f"Tatoeba files not found at {tatoeba_dir}")
            return []
        
        with open(en_file, 'r', encoding='utf-8') as f:
            en_sentences = [line.strip() for line in f.readlines()]
        
        with open(pes_file, 'r', encoding='utf-8') as f:
            pes_sentences = [line.strip() for line in f.readlines()]
        
        data = []
        for idx, (pes, eng) in enumerate(zip(pes_sentences, en_sentences)):
            if pes and eng:
                data.append({
                    'id': f"tatoeba_{idx}",
                    'source_text': pes,
                    'target_text': eng,
                    'source_lang': 'pes',
                    'target_lang': 'eng',
                    'source': 'tatoeba',
                    'split': 'test' if idx % 10 == 0 else 'train',
                    'variant': 'persian'
                })
        
        logger.info(f"Processed {len(data)} Tatoeba samples")
        return data
    
    # ==================== NLI ====================
    
    def _process_nli_data(self):
        """Process FarsTail NLI dataset."""
        logger.info("Processing NLI dataset (FarsTail)...")
        
        nli_data = self._load_farstail()
        
        if nli_data:
            save_jsonl(nli_data, str(self.processed_dir / 'nli.jsonl'))
            logger.info(f"Saved {len(nli_data)} NLI samples")
    
    def _load_farstail(self) -> List[Dict]:
        """Load FarsTail NLI dataset"""
        nli_dir = self.raw_dir / 'nli'
        
        if not nli_dir.exists():
            logger.warning(f"NLI directory not found at {nli_dir}")
            return []
        
        label_map = {
            'e': 'entailment',
            'n': 'neutral', 
            'c': 'contradiction'
        }
        
        data = []
        
        for split_name, filename in [('train', 'Train-word.csv'), 
                                      ('validation', 'Val-word.csv'),
                                      ('test', 'Test-word.csv')]:
            filepath = nli_dir / filename
            
            if not filepath.exists():
                logger.warning(f"FarsTail {filename} not found")
                continue
            
            try:
                df = pd.read_csv(filepath, sep='\t')
                logger.info(f"Loaded FarsTail {filename} with {len(df)} rows")
                
                for idx, row in df.iterrows():
                    premise = str(row.get('premise', '')).strip()
                    hypothesis = str(row.get('hypothesis', '')).strip()
                    label_val = str(row.get('label', '')).strip().lower()
                    
                    if not premise or not hypothesis:
                        continue
                    
                    if label_val not in label_map:
                        continue
                    
                    data.append({
                        'id': f"farstail_{split_name}_{idx}",
                        'premise': premise,  # Don't normalize yet - preserve variants
                        'hypothesis': hypothesis,  # Don't normalize yet - preserve variants
                        'label': label_map[label_val],
                        'source': 'farstail',
                        'split': split_name if split_name != 'validation' else 'dev',
                        'variant': 'persian_standard'
                    })
                    
            except Exception as e:
                logger.error(f"Error loading FarsTail {filename}: {e}")
        
        logger.info(f"Processed {len(data)} FarsTail samples")
        return data
    
    # ==================== QA ====================
    
    def _process_qa_data(self):
        """Process PQuAD QA dataset."""
        logger.info("Processing QA dataset (PQuAD)...")
        
        qa_data = self._load_pquad()
        
        if qa_data:
            save_jsonl(qa_data, str(self.processed_dir / 'qa.jsonl'))
            logger.info(f"Saved {len(qa_data)} QA samples")
    
    def _load_pquad(self) -> List[Dict]:
        """Load PQuAD dataset"""
        qa_dir = self.raw_dir / 'qa' / 'pquad'
        
        if not qa_dir.exists():
            logger.warning(f"PQuAD directory not found at {qa_dir}")
            return []
        
        data = []
        
        for split_name, filename in [('train', 'Train.json'),
                                      ('test', 'Test.json'),
                                      ('validation', 'Validation.json')]:
            filepath = qa_dir / filename
            
            if not filepath.exists():
                logger.warning(f"PQuAD {filename} not found")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    pquad_data = json.load(f)
                
                for article in pquad_data.get('data', []):
                    title = article.get('title', '')
                    
                    for para in article.get('paragraphs', []):
                        context = para.get('context', '').strip()
                        
                        for qa in para.get('qas', []):
                            qid = qa.get('id', '')
                            question = qa.get('question', '').strip()
                            is_impossible = qa.get('is_impossible', False)
                            
                            answers = []
                            for ans in qa.get('answers', []):
                                ans_text = ans.get('text', '').strip()
                                if ans_text:
                                    answers.append(ans_text)
                            
                            if is_impossible or not answers:
                                continue
                            
                            data.append({
                                'id': f"pquad_{qid}",
                                'context': context,  # Don't normalize yet - preserve variants
                                'question': question,  # Don't normalize yet - preserve variants
                                'answers': answers,
                                'title': title,
                                'source': 'pquad',
                                'split': split_name if split_name != 'validation' else 'dev',
                                'variant': 'persian_standard'
                            })
                
                logger.info(f"Loaded PQuAD {filename}")
                
            except Exception as e:
                logger.error(f"Error loading PQuAD {filename}: {e}")
        
        logger.info(f"Processed {len(data)} PQuAD samples")
        return data
    
    def load_task_data(
        self,
        task: str,
        split: str = 'test',
        variant: Optional[str] = None,
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load processed data for a specific task.
        
        Args:
            task: One of 'sentiment', 'mt', 'nli', 'qa'
            split: Data split ('train', 'test', 'dev')
            variant: Language variant filter (e.g., 'persian', 'dari', 'tajik')
            num_samples: Maximum number of samples to return
                        If None, returns ALL available data (full dataset)
        
        Returns:
            List of data dictionaries
        """
        filepath = self.processed_dir / f'{task}.jsonl'
        
        if not filepath.exists():
            logger.warning(f"Processed data not found: {filepath}")
            logger.info("Run process_all_datasets() first")
            return []
        
        data = load_jsonl(str(filepath))
        
        # Filter by split
        if split:
            data = [d for d in data if d.get('split') == split]
        
        # Filter by variant
        if variant:
            data = [d for d in data if d.get('variant') == variant]
        
        total_available = len(data)
        
        # Limit samples only if explicitly requested
        if num_samples is not None and len(data) > num_samples:
            import random
            # Use deterministic sampling with fixed seed
            random.seed(self.config['evaluation']['seed'])
            data = random.sample(data, num_samples)
            logger.info(
                f"Sampled {num_samples} of {total_available} available samples for {task}/{split}"
            )
        else:
            logger.info(
                f"Loaded complete dataset: {len(data)} samples for {task}/{split}"
            )
        
        return data
    
    def get_data_stats(self) -> Dict:
        """Get statistics about all processed datasets."""
        stats = {}
        
        for task in ['sentiment', 'mt', 'nli', 'qa']:
            filepath = self.processed_dir / f'{task}.jsonl'
            
            if not filepath.exists():
                stats[task] = {'total': 0, 'processed': False}
                continue
            
            data = load_jsonl(str(filepath))
            
            # Count by split
            splits = {}
            variants = {}
            sources = {}
            
            for d in data:
                split = d.get('split', 'unknown')
                splits[split] = splits.get(split, 0) + 1
                
                variant = d.get('variant', 'unknown')
                variants[variant] = variants.get(variant, 0) + 1
                
                source = d.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            stats[task] = {
                'total': len(data),
                'processed': True,
                'by_split': splits,
                'by_variant': variants,
                'by_source': sources
            }
        
        return stats


def process_datasets(config: Dict) -> DataLoader:
    """Main function to process all datasets."""
    loader = DataLoader(config)
    loader.process_all_datasets()
    return loader