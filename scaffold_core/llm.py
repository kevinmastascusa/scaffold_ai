"""
LLM integration module using Hugging Face Transformers.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import sys
import logging
import time

# Import ONNX Runtime components if available
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install optimum[onnxruntime]")

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
    HF_TOKEN,
    USE_ONNX
)
logger.debug("Configuration imported successfully")

class LLMManager:
    def __init__(self):
        """Initialize the LLM pipeline."""
        logger.debug(f"Initializing LLM Manager with model: {LLM_MODEL}")
        logger.debug(f"Using device: {LLM_DEVICE}")
        
        # Try to get token from environment or use a fallback
        self.hf_token = HF_TOKEN or os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN environment variable not set. "
                "Please set it to access the model."
            )

        # --- Robust local model caching logic ---
        # Use HuggingFace cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        logger.debug(f"Using cache directory: {cache_dir}")

        logger.debug("Loading tokenizer...")
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False  # Use slow tokenizer to avoid version compatibility issues
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with default method: {e}")
            logger.debug("Trying alternative tokenizer loading method...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False,  # Use slow tokenizer to avoid v3 tokenizer issues
                legacy=True,  # Use legacy mode for better compatibility
                
            )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.debug("Tokenizer loaded successfully")

        # Check if we should use ONNX Runtime
        if USE_ONNX and ONNX_AVAILABLE:
            logger.debug("Loading model with ONNX Runtime optimization...")
            try:
                start_time = time.time()
                
                # Load model with ONNX Runtime optimization
                model = ORTModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    export=True,  # Export to ONNX format
                    provider="CPUExecutionProvider",  # Use CPU provider
                    token=self.hf_token
                )
                
                # Create pipeline with ONNX model
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_full_text=False  # Only return generated completion
                )
                
                load_time = time.time() - start_time
                logger.debug(f"ONNX optimized model loaded in {load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                logger.warning("Falling back to standard model loading...")
                self._load_standard_model()
        else:
            # Standard model loading
            self._load_standard_model()
    
    def _load_standard_model(self):
        """Load model using standard Hugging Face pipeline."""
        logger.debug("Loading model using standard pipeline approach...")
        
        # Use pipeline for better memory management and compatibility
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=LLM_MODEL,
                tokenizer=self.tokenizer,
                token=self.hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False  # Only return generated completion
            )
            logger.debug("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
        
        # Pipeline is already created above
        logger.debug("Pipeline setup complete")
    
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
            max_new_tokens: Optional override for max response length
            temperature: Optional override for temperature
            top_p: Optional override for top-p sampling
            
        Returns:
            The generated response text
        """
        # Format prompt based on model type
        if "mistral" in LLM_MODEL.lower():
            formatted_prompt = f"[INST] {prompt} [/INST]"
        elif "llama" in LLM_MODEL.lower():
            # Use tokenizer chat template if available
            try:
                messages = [
                    {"role": "system", "content": "You are an expert in sustainability education and engineering curriculum development. Provide clear, detailed answers with citations."},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = f"""<|system|>
You are an expert in sustainability education and engineering curriculum development. Provide clear, detailed answers with citations.
<|user|>
{prompt}
<|assistant|>"""
        else:
            # Fallback format for other models
            formatted_prompt = f"""<|system|>You are a helpful AI assistant.<|endoftext|>
<|user|>{prompt}<|endoftext|>
<|assistant|>"""
        
        try:
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens or 512,
                temperature=temperature or LLM_TEMPERATURE,
                top_p=top_p or LLM_TOP_P,
                do_sample=True
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
            
            # Check for truncation indicators
            truncation_indicators = [
                "...", "etc.", "and so on", "continues", "more", 
                "further", "additionally", "moreover", "furthermore"
            ]
            
            is_truncated = False
            for indicator in truncation_indicators:
                if response_text.lower().endswith(indicator.lower()):
                    is_truncated = True
                    break
            
            # Check if response seems incomplete (ends mid-sentence)
            if response_text and not response_text.endswith(('.', '!', '?', ':', ';')):
                is_truncated = True
            
            if is_truncated:
                logger.warning("Response appears to be truncated - consider increasing max_new_tokens")
                # Add a note about truncation
                response_text += "\n\n[Note: Response may be incomplete due to length limits]"
            
            # Log response statistics
            response_tokens = len(response_text.split())
            logger.info(f"Generated response with {response_tokens} words")
            
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

# Global instance - lazy loaded
_llm_instance = None

def get_llm():
    """Get the global LLM instance, creating it if necessary."""
    global _llm_instance
    if _llm_instance is None:
        # Load environment variables if not already loaded
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        _llm_instance = LLMManager()
    return _llm_instance

# For backward compatibility - don't initialize at import time
llm = None 