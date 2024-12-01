from src.llama_deps import Transformer as Llama3
from src.vit import ViT
from tinygrad.nn.state import get_parameters
from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance


def create_model_and_transforms(
    vision_encoder: ViT,
    lang_encoder: Llama3,
    text_tokenizer,
    cross_attn_every_n_layers: int = 1,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        vision_encoder: pretrained vision encoder
        lang_encoder: pretrained language encoder
        tokenizer: pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    # set the vision encoder to output the visual features
    vision_encoder.output_tokens = True

    class EmbeddingFnMixin:
        def get_input_embeddings(self):
            return self.tok_embeddings

        def set_input_embeddings(self, new_embeddings):
            self.tok_embeddings = new_embeddings

    extend_instance(lang_encoder, EmbeddingFnMixin)
    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)

    total_vocab_size = text_tokenizer.num_base_tokens + len(text_tokenizer.special_tokens)
    lang_encoder.resize_token_embeddings(total_vocab_size)

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad = False
    assert sum(p.numel() for p in get_parameters(model) if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad = True

    for layer in model.lang_encoder.gated_cross_attn_layers:
        if layer is not None:
            layer.requires_grad = True

    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().weight.requires_grad = True
        # TODO: investigate also training the output embeddings when untied

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in get_parameters(model) if p.requires_grad)} trainable parameters"
    )

    return model, text_tokenizer


def _infer_decoder_layers_attr_name(model):
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
    "transformer": "layers",
}
