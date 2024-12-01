import ast
from pathlib import Path
from src import factory
from src import vit
from src import llama3
from tinygrad import Device, Tensor, Context
from tinygrad.helpers import fetch
from PIL import Image
import requests
import numpy as np

from src.flamingo_lm import FlamingoLMMixin
from src.llama_deps import Transformer as Llama3
from src.utils import extend_instance


def process_image(img: Image) -> Tensor:
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize(
        (int(224 * max(aspect_ratio, 1.0)), int(224 * max(1.0 / aspect_ratio, 1.0)))
    )
    img = np.array(img)
    y0, x0 = (np.asarray(img.shape)[:2] - 224) // 2
    img = img[y0 : y0 + 224, x0 : x0 + 224]
    img = np.moveaxis(img, [2, 0, 1], [0, 1, 2])
    img = img.astype(np.float32)[:3].reshape(1, 3, 224, 224)
    img /= 255.0
    img -= 0.5
    img /= 0.5
    return Tensor(img)


def get_llama_response(
    model: Llama3, tokenizer: llama3.Tokenizer, prompt: str, max_tokens: int = 100
):
    # Encode the prompt
    input_tokens = tokenizer.encode(prompt, allow_special=True)

    # Keep track of all tokens (prompt + generated)
    all_tokens = input_tokens.copy()
    start_pos = len(input_tokens) - 1  # Start position is end of prompt

    # Generate tokens one at a time
    token_count = 0
    while token_count < max_tokens:
        # Get next token prediction using full context
        next_token = model(
            Tensor([all_tokens]),  # Pass all tokens as context
            start_pos,
            temperature=0.7,
            top_k=5,
            top_p=0.9,
        ).item()

        # Break if we hit a stop token
        if next_token in tokenizer.stop_tokens:
            break

        all_tokens.append(next_token)
        start_pos += 1
        token_count += 1

    # Decode and return only the new tokens (the response)
    response_tokens = all_tokens[len(input_tokens) :]
    response = tokenizer.decode(response_tokens)
    return response


if __name__ == "__main__":
    print(f"using {Device.DEFAULT} backend")
    vision_encoder = vit.ViT(embed_dim=192, num_heads=3)  # Tiny ViT
    vision_encoder.load_from_pretrained()

    fetch(
        "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
        "tokenizer.model",
        subdir="llama3-1b-instruct",
    )
    model_path = fetch(
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
        "Llama-3.2-1B-Instruct-Q6_K.gguf",
        subdir="llama3-1b-instruct",
    )

    tokenizer = llama3.Tokenizer(
        f"{model_path if model_path.is_dir() else model_path.parent}/tokenizer.model",
        # add_special_tokens=["<|endofchunk|>", "<image>", "<PAD>"],
    )

    llama3_model = llama3.build_transformer(
        model_path, model_size="1B", quantize=None, device=Device.DEFAULT
    )

    # prompt = "The capital of France is "
    # response = get_llama_response(llama3_model, tokenizer, prompt, max_tokens=10)
    # print(f"Prompt: {prompt}")
    # print(f"Response: {response}")

    model, tokenizer = factory.create_model_and_transforms(
        vision_encoder=vision_encoder,
        lang_encoder=llama3_model,
        text_tokenizer=tokenizer,
        cross_attn_every_n_layers=1,
    )

    """
    Step 1: Load images
    """
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
        ).raw
    )

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape
    batch_size x num_media x num_frames x channels x height x width.
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    vision_x = [
        np.expand_dims(process_image(demo_image_one), axis=0),
        np.expand_dims(process_image(demo_image_two), axis=0),
        np.expand_dims(process_image(query_image), axis=0),
    ]

    vision_x = np.concatenate(vision_x, axis=0)
    vision_x = np.expand_dims(vision_x, axis=0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text
    portion associated with an image.
    """
    tokenizer.padding_side = (
        "left"  # For generation padding tokens should be on the left
    )
    lang_x = tokenizer.encode(
        "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of",
        allow_special=True,
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=Tensor(vision_x),
        lang_x=Tensor(lang_x),
        max_new_tokens=20,
        num_beams=3,
    )

    print("Generated text: ", tokenizer.decode(generated_text[0]))
