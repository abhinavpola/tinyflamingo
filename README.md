# TinyFlamingo

A tinygrad implementation of the Flamingo model. Pretty much just plugged tiny ViT and tiny llama together.

```
pip install -r requirements.txt
python demo.py
```

Sources:
- https://github.com/mlfoundations/open_flamingo
- https://github.com/tinygrad/tinygrad/blob/master/examples/llama3.py
- https://github.com/tinygrad/tinygrad/blob/master/examples/vit.py
- https://github.com/lucidrains/einops-exts

## TODO
* Try it with bigger llama
* Make it trainable
* Write an eval script
* Get it to work with other vision-language pairings
* Optimize
* Clean up