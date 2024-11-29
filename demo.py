from pathlib import Path
from src import factory
from src import vit
from src import llama3
from tinygrad import Device, Tensor
from tinygrad.helpers import fetch
from PIL import Image
import requests
import numpy as np

if __name__ == "__main__":
    Tensor.no_grad = True
    print(f"using {Device.DEFAULT} backend")
    vision_encoder = vit.ViT(embed_dim=192, num_heads=3)
    vision_encoder.load_from_pretrained()

    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
    model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")


    tokenizer = llama3.Tokenizer(str((model_path if model_path.is_dir() else model_path.parent) / "tokenizer.model"))

    llama3_model = llama3.build_transformer(model_path, model_size="1B", quantize="int8")

    model, image_processor, tokenizer = factory.create_model_and_transforms(
        vision_encoder=vision_encoder,
        lang_encoder=llama3_model,
        tokenizer=tokenizer,
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
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
            stream=True
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
    np.expand_dims(image_processor(demo_image_one), axis=0),
    np.expand_dims(image_processor(demo_image_two), axis=0),
    np.expand_dims(image_processor(query_image), axis=0)
]

    vision_x = np.concatenate(vision_x, axis=0)
    vision_x = np.expand_dims(vision_x, axis=0)

    print(vision_x.shape)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
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


