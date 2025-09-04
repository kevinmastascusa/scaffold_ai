"""
LLM integration module using Hugging Face Transformers.
"""

import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

# Import ONNX Runtime components if available
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning(
        "ONNX Runtime not available. Install with: "
        "pip install optimum[onnxruntime]"
    )

# Detect available ORT providers (CUDA vs CPU)
try:
    import onnxruntime as ort  # type: ignore
    AVAILABLE_ORT_PROVIDERS = set(ort.get_available_providers())
except Exception:
    AVAILABLE_ORT_PROVIDERS = set()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CPU-specific optimizations
if torch.cuda.is_available():
    logger.debug("CUDA available - using GPU optimizations")
else:
    logger.debug("CUDA not available - applying CPU optimizations")
    # Enable CPU optimizations for faster inference
    torch.set_num_threads(6)  # Increased from 4 to 6
    torch.backends.cudnn.benchmark = False  # Disable CUDA optimizations
    torch.backends.cudnn.deterministic = True  # Deterministic CPU
    # Add memory optimization
    torch.backends.cudnn.enabled = False  # Disable cuDNN for CPU-only usage

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
    USE_ONNX,
    get_dynamic_temperature,
    get_dynamic_top_p,
    ENABLE_TRUNCATION_DETECTION,
    get_dynamic_max_new_tokens,
)
logger.debug("Configuration imported successfully")

# Global singleton instance for model caching
_llm_manager_instance = None

class LLMManager:
    def __init__(self):
        """Initialize the LLM pipeline."""
        logger.debug(f"Initializing LLM Manager with model: {LLM_MODEL}")
        logger.debug(f"Using device: {LLM_DEVICE}")

        # Check if we already have a cached instance
        global _llm_manager_instance
        if _llm_manager_instance is not None:
            logger.debug("Using cached LLM Manager instance")
            # Copy attributes from cached instance
            self.tokenizer = _llm_manager_instance.tokenizer
            self.pipeline = _llm_manager_instance.pipeline
            self.hf_token = _llm_manager_instance.hf_token
            return

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

        # Prefer local cached tokenizer next to ONNX export for fast startups
        onnx_cache_dir = os.path.join(
            project_root, "outputs", "onnx_models", LLM_MODEL.replace("/", "__")
        )
        tokenizer_source = (
            onnx_cache_dir if os.path.exists(os.path.join(onnx_cache_dir, "tokenizer.json"))
            or os.path.exists(os.path.join(onnx_cache_dir, "vocab.json"))
            else LLM_MODEL
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False,
                local_files_only=os.path.isdir(tokenizer_source)
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with default method: {e}")
            logger.debug("Trying alternative tokenizer loading method...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False,
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

                # Configure ONNX Runtime session options for optimal threading
                session_options = ort.SessionOptions()
                use_cuda = (
                    torch.cuda.is_available()
                    and "CUDAExecutionProvider" in AVAILABLE_ORT_PROVIDERS
                )

                if use_cuda:
                    # GPU-optimized settings for maximum performance
                    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    session_options.enable_mem_pattern = True
                    session_options.enable_cpu_mem_arena = False  # Disable for GPU
                    providers = [
                        "CUDAExecutionProvider", "CPUExecutionProvider"
                    ]
                    provider_options = [
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 10 * 1024 * 1024 * 1024,  # 10GB
                            "cudnn_conv_use_max_workspace": "1",
                            "do_copy_in_default_stream": "1"
                        },
                        {},
                    ]
                    logger.info("🚀 Using GPU-optimized ONNX Runtime settings")
                else:
                    # Optimized CPU threading
                    session_options.intra_op_num_threads = 6
                    session_options.inter_op_num_threads = 1
                    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                    session_options.enable_mem_pattern = True
                    session_options.enable_cpu_mem_arena = True
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    providers = ["CPUExecutionProvider"]
                    provider_options = [{}]

                # Prepare persistent ONNX cache to avoid re-export on every start
                onnx_cache_dir = os.path.join(
                    project_root, "outputs", "onnx_models", LLM_MODEL.replace("/", "__")
                )
                os.makedirs(onnx_cache_dir, exist_ok=True)

                # Detect pre-exported model
                try:
                    existing_files = os.listdir(onnx_cache_dir)
                except Exception:
                    existing_files = []
                has_onnx = any(fname.endswith(".onnx") for fname in existing_files)
                has_config = os.path.exists(os.path.join(onnx_cache_dir, "config.json"))
                preexport_exists = has_onnx and has_config

                if preexport_exists:
                    logger.debug(
                        f"Loading pre-exported ONNX model from cache: {onnx_cache_dir}"
                    )
                    model = ORTModelForCausalLM.from_pretrained(
                        onnx_cache_dir,
                        providers=providers,
                        session_options=session_options,
                        provider_options=provider_options,
                    )
                else:
                    logger.debug("Exporting model to ONNX (first run) — this may take a while...")
                    model = ORTModelForCausalLM.from_pretrained(
                        LLM_MODEL,
                        export=True,
                        token=self.hf_token,
                        session_options=session_options,
                        providers=providers,
                        provider_options=provider_options,
                    )
                    # Persist export for subsequent fast startups
                    try:
                        model.save_pretrained(onnx_cache_dir)
                        logger.debug(f"Saved ONNX model to cache: {onnx_cache_dir}")
                        # Also save tokenizer alongside for fully offline startup
                        try:
                            self.tokenizer.save_pretrained(onnx_cache_dir)
                        except Exception as tok_err:
                            logger.warning(
                                f"Could not save tokenizer to ONNX cache: {tok_err}"
                            )
                    except Exception as save_err:
                        logger.warning(f"Could not save ONNX model cache: {save_err}")

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
            # CPU-optimized pipeline configuration
            pipeline_kwargs = {
                "task": "text-generation",
                "model": LLM_MODEL,
                "tokenizer": self.tokenizer,
                "token": self.hf_token,
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "return_full_text": False,  # Only return generated completion
            }

            # Device-specific optimizations
            if LLM_DEVICE == "cpu":
                pipeline_kwargs.update({
                    "torch_dtype": torch.float32,  # Use float32 for CPU
                    "device": "cpu",
                })
                logger.info("💻 Using CPU-optimized pipeline settings")
            else:
                # GPU-optimized settings
                pipeline_kwargs.update({
                    "torch_dtype": torch.float16,  # Use float16 for GPU memory efficiency
                    "device": "cuda:0",  # Explicitly use first GPU
                })
                logger.info("🚀 Using GPU-optimized pipeline settings")

            self.pipeline = pipeline(**pipeline_kwargs)
            logger.debug("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

        # Pipeline is already created above
        logger.debug("Pipeline setup complete")

        # Cache this instance globally
        global _llm_manager_instance
        _llm_manager_instance = self
        logger.debug("LLM Manager instance cached for reuse")

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
        # Format prompt based on model type, but avoid double-wrapping if already formatted
        lower_model = LLM_MODEL.lower()
        if "mistral" in lower_model or "mixtral" in lower_model:
            if "[INST]" in prompt:
                formatted_prompt = prompt
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "llama" in lower_model:
            if "<|user|>" in prompt or "<|system|>" in prompt:
                formatted_prompt = prompt
            else:
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
            # Generate response with dynamic temperature
            current_temperature = temperature or get_dynamic_temperature()
            current_top_p = top_p or get_dynamic_top_p()

            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens or get_dynamic_max_new_tokens() or LLM_MAX_NEW_TOKENS,
                temperature=current_temperature,
                top_p=current_top_p,
                top_k=50,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
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

            # Clean up response formatting
            response_text = response_text.strip()

            # Remove common formatting artifacts
            artifacts_to_remove = [
                "Response:", "Answer:", "Assistant:", "AI:",
                "<|assistant|>", "<|user|>", "<|system|>",
                "[/INST]", "</s>", "<|endoftext|>"
            ]

            for artifact in artifacts_to_remove:
                response_text = response_text.replace(artifact, "").strip()

            # Ensure proper bullet point formatting
            response_text = response_text.replace("•", "•").replace("·", "•")

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

            if is_truncated and ENABLE_TRUNCATION_DETECTION:
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

    def generate_response_with_continuation(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate a response with automatic continuation if truncated.

        Args:
            prompt: The input prompt
            max_new_tokens: Optional override for max response length
            temperature: Optional override for temperature
            top_p: Optional override for top-p sampling

        Returns:
            Complete response text (potentially with continuation)
        """
        try:
            # Generate initial response
            response_text = self.generate_response(prompt, max_new_tokens, temperature, top_p)

            # Check if continuation is needed
            if ENABLE_TRUNCATION_DETECTION and self._is_response_truncated(response_text):
                logger.info("Detected truncated response, generating continuation...")
                continuation = self._generate_continuation(
                    prompt, response_text, max_new_tokens, temperature, top_p
                )
                if continuation and continuation.strip():
                    # Remove the truncation note if it exists
                    if "[Note: Response may be incomplete due to length limits]" in response_text:
                        response_text = response_text.replace("\n\n[Note: Response may be incomplete due to length limits]", "")

                    # Combine responses with better formatting
                    response_text = response_text.rstrip() + " " + continuation.strip()
                    logger.info("Successfully generated continuation")

            # Log final response statistics
            response_tokens = len(response_text.split())
            logger.info(f"Generated complete response with {response_tokens} words")

            return response_text

        except Exception as e:
            logger.error(f"Error during enhanced generation: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _is_response_truncated(self, response_text: str) -> bool:
        """Check if a response appears to be truncated."""
        # Check for truncation indicators
        truncation_indicators = [
            "...", "etc.", "and so on", "continues", "more",
            "further", "additionally", "moreover", "furthermore",
            "[Note: Response may be incomplete due to length limits]"
        ]

        for indicator in truncation_indicators:
            if indicator.lower() in response_text.lower():
                return True

        # Check if response seems incomplete (ends mid-sentence)
        if response_text and not response_text.strip().endswith(('.', '!', '?', ':', ';')):
            return True

        return False

    def _generate_continuation(
        self,
        original_prompt: str,
        partial_response: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate a continuation for a truncated response."""
        try:
            # Clean up partial response for continuation prompt
            clean_partial = partial_response.replace("\n\n[Note: Response may be incomplete due to length limits]", "").strip()

            # Get the last few sentences to provide context
            sentences = clean_partial.split('.')
            if len(sentences) > 3:
                context = '. '.join(sentences[-3:]).strip()
            else:
                context = clean_partial

            # Create continuation prompt
            continuation_prompt = f"""Please continue and complete this response about: "{original_prompt}"

The response so far ends with: "{context}"

Continue from where it left off and provide a complete conclusion:"""

            logger.debug("Generating continuation with context")

            # Generate continuation with shorter length to avoid infinite recursion
            continuation_tokens = min(max_new_tokens or LLM_MAX_NEW_TOKENS, 500)

            # Use original generate_response to avoid recursion
            continuation = self.generate_response(
                continuation_prompt,
                max_new_tokens=continuation_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Clean up continuation (remove any repetition of context)
            if continuation and len(continuation.strip()) > 20:
                # Remove the truncation note from continuation if present
                continuation = continuation.replace("\n\n[Note: Response may be incomplete due to length limits]", "")
                return continuation.strip()

            return ""

        except Exception as e:
            logger.error(f"Error generating continuation: {str(e)}")
            return ""

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