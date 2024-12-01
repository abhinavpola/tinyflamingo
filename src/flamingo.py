from einops import rearrange

from .helpers import PerceiverResampler
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from .vit import ViT
from .llama3 import Transformer


class Flamingo:
    def __init__(
        self,
        vision_encoder: ViT,
        lang_encoder: Transformer,
        eoc_token_id: int,
        media_token_id: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        requires_grad: bool = False,
    ):
        """
        Args:
            vision_encoder: Tinygrad ViT
            lang_encoder: Tinygrad LLaMa
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vision_encoder.embed_dim
        self.lang_dim = lang_encoder.dim

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
        if not self.requires_grad:
            with Tensor.no_grad():
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        vision_x: Tensor,
        lang_x: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (Tensor, optional): Attention mask. Defaults to None.
            labels (Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """

        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: Tensor,
        lang_x: Tensor,
        attention_mask: Tensor = None,
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
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder(lang_x, 0)
        # output = self.lang_encoder.generate(
        #     input_ids=lang_x,
        #     attention_mask=attention_mask,
        #     eos_token_id=eos_token_id,
        #     num_beams=num_beams,
        #     **kwargs,
        # )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
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
        vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def _condition_media_locations(self, input_ids: Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)
