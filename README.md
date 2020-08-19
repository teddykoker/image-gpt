# Image GPT

PyTorch implementation of Image GPT, based on paper *Generative Pretraining from Pixels* [(Chen et al.)](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
and accompanying [code](https://github.com/openai/image-gpt).

<img src="figures/mnist.png" height="256px"/> <img src="figures/fmnist.png" height="256px"/>
<br>
*Model-generated completions of half-images from test set. First column is
input; last column is original image*

Differences from original paper:
 * ~~Uses 4-bit grayscale images instead of 9-bit RGB~~
 * ~~28x28 images are used instead of 32x32~~
 * ~~Quantization is done naively using division, not KNN~~
 * *k*-Means is now used for image quantization, number of centroids can be chosen in the
     `prepare_data` command line script.

According to their [blog post](https://openai.com/blog/image-gpt/), the largest
model, iGPT-L (1.4 M parameters), was trained for 2500 V100-days. By greatly reducing the number of
attention head, number of layers, and input size (which effects model size
quadratically), we can train our own model (26 K parameters) on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) on a single
NVIDIA 2070 in less than 2 hours.

- [Image GPT](#image-gpt)
  * [Usage](#usage)
    + [Pre-trained Models](#pre-trained-models)
    + [Compute Centroids](#compute-centroids)
    + [Training](#training)
      - [Generative Pre-training](#generative-pre-training)
      - [Classification Fine-tuning](#classification-fine-tuning)
    + [Sampling](#sampling)

## Usage

### Pre-trained Models

Pre-trained models are located in `models` directory.

### Compute Centroids

Images are downloaded, and centroids are computed using *k*-means with
`num_clusters` clusters.  These centroids are used to quantize the images before
they are fed into the model.

```bash
# options: mnist, fmnist, cifar10
python src/compute_centroids.py --dataset mnist --num_clusters=8

# creates data/<dataset>_centroids.npy
```

*Note: Use the same `num_clusters` as `num_vocab` in your model*.

### Training

Models can be trained using `src/run.py` with the `train` subcommand. 

#### Generative Pre-training

```bash
python src/run.py \
    --dataset mnist \
    --centroids data/mnist_centroids.npy \
    train \
    --name mnist_gen \
```

The following hyperparameters can also be provided. Smallest model from paper is
shown for comparison.

Argument          | Default  | iGPT-S [(Chen et al.)](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
---               | ---      | ---
`--embed_dim`     | 16       | 512
`--num_heads`     | 2        | 8
`--num_layers`    | 8        | 24
`--num_pixels`    | 28       | 32
`--num_vocab`     | 16       | 512
`--batch_size`    | 64       | 128
`--learning_rate` | 0.01     | 0.01
`--steps`         | 25000    | 1000000

#### Classification Fine-tuning

Pre-trained models can be fine-tuned by passing the path to the pre-trained
checkpoint to `--pretrained`, along with the `--classify` argument. I have found
a small reduction in learning rate is necessary.

```bash
python src/run.py train \
    --name mnist_clf  \
    --pretrained models/mnist_gen.ckpt \
    --classify \
    --learning_rate 3e-3
```

### Sampling 

Figures like those seen above can be created using random images from
test set:

```bash
# outputs to figure.png
python src/sample.py models/mnist_gen.ckpt
```

Gifs like the one seen in [my tweet](https://twitter.com/teddykoker/status/1275809619705806850) can be made
like so:

```bash
# outputs to out.gif
python src/gif.py models/mnist_gen.ckpt
```
