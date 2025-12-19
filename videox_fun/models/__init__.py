import importlib.util

from diffusers import AutoencoderKL
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection, LlamaModel,
                          LlamaTokenizerFast, LlavaForConditionalGeneration,
                          Mistral3ForConditionalGeneration, PixtralProcessor,
                          Qwen3ForCausalLM, T5EncoderModel, T5Tokenizer,
                          T5TokenizerFast)

try:
    from transformers import (Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Qwen2Tokenizer, Qwen2VLProcessor)
except:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    Qwen2VLProcessor, Qwen2_5_VLConfig = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")

from .flux2_image_processor import Flux2ImageProcessor
from .flux2_transformer2d import Flux2Transformer2DModel
from .flux2_transformer2d_control import Flux2ControlTransformer2DModel
from .flux2_vae import AutoencoderKLFlux2
from .flux_transformer2d import FluxTransformer2DModel
from .qwenimage_transformer2d import QwenImageTransformer2DModel
from .qwenimage_vae import AutoencoderKLQwenImage
from .z_image_transformer2d import ZImageTransformer2DModel
from .z_image_transformer2d_control import ZImageControlTransformer2DModel