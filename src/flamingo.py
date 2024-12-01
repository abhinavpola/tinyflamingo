from tinygrad import Tensor
from einops import rearrange
from .helpers import PerceiverResampler

from .vit import ViT
from .llama3 import Transformer as Llama3


class Flamingo:
    def __init__(
        self,
        vision_encoder: ViT,
        lang_encoder: Llama3,
        eoc_token_id: int,
        media_token_id: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        requires_grad: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.lang_dim = lang_encoder.dim
        self.vis_dim = vision_encoder.embed_dim
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing
        self.requires_grad = requires_grad

    def __call__(self, *args, **kwargs):
        with Tensor.train(self.requires_grad):
            return self._forward(*args, **kwargs)

    def _forward(
        self,
        vision_x: Tensor,
        lang_x: Tensor,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.8,
        alpha_f: float = 0.0,
        alpha_p: float = 0.0,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (Tensor): Language input ids
                shape (B, T_txt)
            labels (Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the forward pass is completed.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            top_k (int, optional): Top-k sampling. Defaults to 0.
            top_p (float, optional): Top-p sampling. Defaults to 0.8.
            alpha_f (float, optional): Alpha frequency. Defaults to 0.0.
            alpha_p (float, optional): Alpha presence. Defaults to 0.0.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."
        assert vision_x is not None, "Must provide vision_x."

        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=lang_x)

        # Get the start position for the language model
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]

        # Call the Llama3 transformer with its expected interface
        output = self.lang_encoder(
            tokens=lang_x,
            start_pos=start_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: Tensor,
        lang_x: Tensor,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            Tensor: lang_x with generated tokens appended to it
        """
        with Tensor.test():
            num_beams = kwargs.pop("num_beams", 1)
            if num_beams > 1:
                vision_x = vision_x.repeat_interleave(num_beams, dim=0)

            self._encode_vision_x(vision_x=vision_x)

            eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)

            assert lang_x.ndim == 2, "lang_x should be of shape (b, t)"
            output = self.lang_encoder.generate(
                input_ids=lang_x,
                eos_token_id=eos_token_id,
                num_beams=num_beams,
                **kwargs,
            )

            self.lang_encoder.clear_conditioned_layers()
            return output

    def _encode_vision_x(self, vision_x: Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with Tensor.test():
            vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
