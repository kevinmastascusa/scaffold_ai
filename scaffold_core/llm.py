"""
LLM integration module using Hugging Face Transformers.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.debug("Starting LLM module initialization...")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
logger.debug(f"Added project root to Python path: {project_root}")

logger.debug("Importing configuration...")
from scaffold_core.config import (
    LLM_MODEL,
    LLM_TASK,
    LLM_DEVICE,
    LLM_MAX_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_BATCH_SIZE,
    LLM_LOAD_IN_8BIT,
    LLM_LOAD_IN_4BIT,
    HF_TOKEN
)
logger.debug("Configuration imported successfully")

class LLMManager:
    def __init__(self):
        """Initialize the LLM pipeline."""
        logger.debug(f"Initializing LLM Manager with model: {LLM_MODEL}")
        logger.debug(f"Using device: {LLM_DEVICE}")
        
        if not HF_TOKEN:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN environment variable not set. "
                "Please set it to access the model."
            )

        # --- Robust local model caching logic ---
        # Convert model name to a safe local directory name for cache
        local_model_dir = os.path.join(
            "models",
            LLM_MODEL.replace("/", "_").replace("-", "_")
        )
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir, exist_ok=True)
            logger.info(f"Ensured local model cache directory exists: {local_model_dir}")

        logger.debug("Loading tokenizer...")
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=HF_TOKEN,
                trust_remote_code=True,
                cache_dir=local_model_dir,
                use_fast=False  # Use slow tokenizer to avoid version compatibility issues
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with default method: {e}")
            logger.debug("Trying alternative tokenizer loading method...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=HF_TOKEN,
                trust_remote_code=True,
                cache_dir=local_model_dir,
                use_fast=False,  # Use slow tokenizer to avoid v3 tokenizer issues
                legacy=True  # Use legacy mode for better compatibility
            )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.debug("Tokenizer loaded successfully")

        logger.debug("Loading model...")
        from transformers import AutoModelForCausalLM
        
        # Configure quantization for faster loading and inference
        if LLM_LOAD_IN_4BIT and LLM_DEVICE == "cuda" and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.debug("Using 4-bit quantization")
        elif LLM_LOAD_IN_8BIT and LLM_DEVICE == "cuda" and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            logger.debug("Using 8-bit quantization")
        else:
            quantization_config = None
            logger.debug("Using full precision (no quantization)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32,
            device_map="auto" if LLM_DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            use_cache=True,
            cache_dir=local_model_dir
        )
        logger.debug("Model loaded successfully")
        
        logger.debug("Creating pipeline...")
        self.pipeline = pipeline(
            LLM_TASK,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            batch_size=LLM_BATCH_SIZE,
            trust_remote_code=True,
            # Performance optimizations
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False  # Only return generated text, not input
        )
        logger.debug("Pipeline created successfully")
    
    def generate_response(
        self, 
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            max_length: Optional override for max response length
            temperature: Optional override for temperature
            top_p: Optional override for top-p sampling
            
        Returns:
            The generated response text
        """
        # Format prompt for Mistral's chat format
        if "mistral" in LLM_MODEL.lower():
            formatted_prompt = f"[INST] {prompt} [/INST]"
        else:
            # Fallback format for other models
            formatted_prompt = f"""<|system|>You are a helpful AI assistant.<|endoftext|>
<|user|>{prompt}<|endoftext|>
<|assistant|>"""
        
        try:
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens or LLM_MAX_NEW_TOKENS,
                temperature=temperature or LLM_TEMPERATURE,
                top_p=top_p or LLM_TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=True
            )
            
            # Extract response text
            if isinstance(outputs, list) and len(outputs) > 0:
                response = outputs[0]
                if isinstance(response, dict) and "generated_text" in response:
                    response_text = response["generated_text"]
                else:
                    response_text = str(response)
            else:
                response_text = str(outputs)
                
            # Extract just the assistant's response
            if "mistral" in LLM_MODEL.lower():
                # For Mistral, extract response after [/INST]
                if "[/INST]" in response_text:
                    response_text = response_text.split("[/INST]")[-1].strip()
                # Remove any remaining special tokens
                response_text = response_text.replace("</s>", "").strip()
            else:
                # For other models
                if "<|assistant|>" in response_text:
                    response_text = response_text.split("<|assistant|>")[-1].strip()
                if "<|endoftext|>" in response_text:
                    response_text = response_text.split("<|endoftext|>")[0].strip()
            return response_text
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return f"Error generating response: {str(e)}"

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Optional override for max new tokens to generate
            temperature: Optional override for temperature
            top_p: Optional override for top-p sampling
            
        Returns:
            List of generated response texts
        """
        return [
            self.generate_response(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            for prompt in prompts
        ]

# Global instance
llm = LLMManager() 