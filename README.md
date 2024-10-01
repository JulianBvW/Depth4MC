# Depth4MC
Depth estimator machine learning model for the sandbox game Minecraft

## Installation

The model itself only needs PyTorch and Numpy as it is a U-Net modelled from scratch.
If you want to create your own dataset, the Python package `mcpi` is used. For that, just create a new Conda environment from the `environment.yml`:
```
conda env create --file environment.yml
```

If you want to use the model, you can simply install it by cloning this repository and in it, type `pip install -e .`.
This will allow you to use the model like this:
```python
from depth4mc.model.D4MCModel import D4MCModel
from depth4mc.dataset.D4MCDataset import DEFAULT_TRANSFORM
```

## Making your own dataset

The dataset I trained on is available on HuggingFace: https://huggingface.co/datasets/JulianBvW/Minecraft-Depth-Images

If you want to create your own dataset just read the `depth4mc/dataset/dataset_maker/README.md`.
It will guide you through how to create a Minecraft server with the right plugins, setup Minecraft and the depth shader, record screenshots, and convert the files into a dataset.

There are good reasons to create a new dataset, as there are some problems with the one I uploaded to HuggingFace:
- The depth recordings stop at around 90 blocks. By playing with the clipping value as described in the `dataset_maker/README.md` one could mitigate this.
- Very rarely, two consecutive screenshots are flipped, meanding image `i` has the depth labels from image `i+1`. This is a recording error from my PC.
- Not enough diversity. The dataset only contains overworld images of a few structures and biomes. Nether or The End are for example completely ignored.
- PyTorch's `transforms.Normalize()` has values from an older version of the dataset and needs to be recalculated.
- No entities. While recording, I disabled all entities and just focused on blocks.

## Training the model

After you have your dataset, you can train the model using the `depth4mc/training/train.py` script.
Type `python depth4mc/training/train.py -h` to get information on arguments like the epoch count or learning rate.

The train code will save results into `depth4mc/training/results`:
- The `losses.csv` files shows you the train and test loss over the epochs.
- The `chekcpoints/` directory saves the model checkpoints every 10 epochs and the final checkpoint as `model_final.pth`.

## Using the model

Pretrained weights can be loaded from HuggingFace: https://huggingface.co/JulianBvW/Depth4MC-Minecraft-Depth-Estimation

Check `depth4mc/comparing/compare.ipynb` or `depth4mc/comparing/evaluate.py` for examples on how to use the model.
Here it is also compared to the [DepthAnything](https://github.com/LiheYoung/Depth-Anything) model.
The `depth4mc/comparing/README.md` instructs on how to use the DepthAnything model.


