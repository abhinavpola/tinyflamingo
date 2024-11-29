from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
import open_clip

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance
from .vit import ViT
from PIL import Image
import numpy as np
from .llama3 import Transformer, Tokenizer
from tinygrad.nn.state import get_parameters

def create_model_and_transforms(
    vision_encoder: ViT,
    lang_encoder: Transformer,
    tokenizer: Tokenizer,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    **flamingo_kwargs,
):
    def image_processor(x):
        img = x
        aspect_ratio = img.size[0] / img.size[1]
        img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
        img = np.array(img)
        y0,x0=(np.asarray(img.shape)[:2]-224)//2
        img = img[y0:y0+224, x0:x0+224]
        img = np.moveaxis(img, [2,0,1], [0,1,2])
        img = img.astype(np.float32)[:3].reshape(1,3,224,224)
        img /= 255.0
        img -= 0.5
        img /= 0.5
        return img
    # set the vision encoder to output the visual features
    # vision_encoder.visual.output_tokens = True

    text_tokenizer = tokenizer
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.special_tokens.extend(
        ["<|endofchunk|>", "<image>", "<PAD>"]
    )
    text_tokenizer.finish_init()

    # if text_tokenizer.pad_token is None:
    #     # Issue: GPT models don't have a pad token, which we use to
    #     # modify labels for the loss.
    #     text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    # lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>", allow_special=True)[-1],
        text_tokenizer.encode("<image>", allow_special=True)[-1],
        vis_dim=vision_encoder.embed_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in get_parameters(model) if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    print(f"model.__class__.__name__: {model.__class__.__name__}")
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "Transformer": "layers",
}