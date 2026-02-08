"""
Model loading and inference for Persian LLM Evaluation
5 Open-Weight Multilingual Models
"""

import torch
from typing import Dict, List, Optional, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from tqdm import tqdm
import logging

from src.utils import get_device, format_prompt

logger = logging.getLogger(__name__)


class MultilingualLLM:
    """
    Wrapper class for multilingual LLM inference.
    Supports the 5 open-weight models specified in config.
    """
    
    def __init__(
        self,
        model_name: str,
        hf_id: str,
        quantization: str = "4bit",
        max_new_tokens: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.hf_id = hf_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.device = device or get_device()
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model with appropriate quantization."""
        logger.info(f"Loading {self.model_name} from {self.hf_id}...")
        
        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        # If "none" or other, no quantization
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype based on model
        # Gemma-3 works better with bfloat16
        if 'gemma-3' in self.hf_id.lower() or 'gemma3' in self.hf_id.lower():
            torch_dtype = torch.bfloat16
            logger.info(f"Using bfloat16 for Gemma-3 model")
        else:
            torch_dtype = torch.float16
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                **model_kwargs
            )
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response for a single prompt.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response (only new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        temperature: float = 0.0,
        do_sample: bool = False,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        """
        responses = []
        
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating with {self.model_name}")
        
        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            
            for prompt in batch_prompts:
                try:
                    response = self.generate(
                        prompt,
                        temperature=temperature,
                        do_sample=do_sample
                    )
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    responses.append("")
        
        return responses
    
    def __del__(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()


class ModelManager:
    """
    Manager class for handling multiple models.
    """
    
    # Model specifications for the 5 open-weight multilingual models
    MODEL_SPECS = {
    "Qwen2.5-1.5B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Alibaba Qwen2.5 instruction-tuned causal language model (1.5B parameters)"
    },
    "Qwen2.5-3B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Alibaba Qwen2.5 instruction-tuned causal language model (3B parameters)"
    },
    "bloomz-1b7": {
        "hf_id": "bigscience/bloomz-1b7",
        "description": "BigScience BLOOMZ multilingual instruction-tuned causal model (1.7B parameters)"
    },
    "bloomz-3b": {
        "hf_id": "bigscience/bloomz-3b",
        "description": "BigScience BLOOMZ multilingual instruction-tuned causal model (3B parameters)"
    },
    "gemma-3-4b-persian": {
        "hf_id": "mshojaei77/gemma-3-4b-persian-v0",
        "description": "Gemma 3 4B causal language model fine-tuned for Persian"
    }
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.loaded_models: Dict[str, MultilingualLLM] = {}
    
    def get_model(self, model_name: str) -> MultilingualLLM:
        """
        Get or load a model by name.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Find model config
        model_config = None
        for m in self.config['models']:
            if m['name'] == model_name:
                model_config = m
                break
        
        if model_config is None:
            raise ValueError(f"Model {model_name} not found in config")
        
        # Load model
        model = MultilingualLLM(
            model_name=model_config['name'],
            hf_id=model_config['hf_id'],
            quantization=model_config.get('quantization', '4bit'),
            max_new_tokens=model_config.get('max_new_tokens', 256)
        )
        
        self.loaded_models[model_name] = model
        return model
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_name}")
    
    def unload_all(self):
        """Unload all models."""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return [m['name'] for m in self.config['models']]
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Dict]:
        """Get information about all supported models."""
        return cls.MODEL_SPECS


def run_inference(
    model: MultilingualLLM,
    task: str,
    data: List[Dict],
    config: Dict
) -> List[Dict]:
    """
    Run inference for a specific task.
    
    Args:
        model: Loaded model instance
        task: Task name ('sentiment', 'mt', 'nli', 'qa')
        data: List of data items
        config: Configuration dictionary
    
    Returns:
        List of results with predictions
    """
    results = []
    task_config = config['tasks'].get(task, {})
    eval_config = config['evaluation']
    
    for item in tqdm(data, desc=f"Running {task} inference"):
        # Format prompt based on task
        if task == 'sentiment':
            prompt = format_prompt(
                task='sentiment',
                text=item['text'],
                labels=task_config.get('labels', ['negative', 'neutral', 'positive'])
            )
        elif task == 'mt':
            prompt = format_prompt(
                task='mt',
                text=item['source_text'],
                source_lang=item['source_lang'],
                target_lang=item['target_lang']
            )
        elif task == 'nli':
            prompt = format_prompt(
                task='nli',
                text=item['premise'],
                context=item['hypothesis'],
                labels=task_config.get('labels', ['entailment', 'neutral', 'contradiction'])
            )
        elif task == 'qa':
            prompt = format_prompt(
                task='qa',
                text=item['question'],
                context=item['context'],
                question=item['question']
            )
        else:
            logger.warning(f"Unknown task: {task}")
            continue
        
        # Generate response
        try:
            response = model.generate(
                prompt,
                temperature=eval_config.get('temperature', 0.0),
                do_sample=eval_config.get('do_sample', False)
            )
        except Exception as e:
            logger.warning(f"Generation failed for item {item['id']}: {e}")
            response = ""
        
        # Store result
        result = {
            'id': item['id'],
            'prompt': prompt,
            'response': response,
            'gold': item.get('label') or item.get('target_text') or item.get('answers', []),
            'variant': item.get('variant', 'unknown'),
            'source': item.get('source', 'unknown')
        }
        results.append(result)
    
    return results