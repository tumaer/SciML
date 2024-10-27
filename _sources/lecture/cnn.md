# Convolutional Neural Networks

## Limitations of MLP

In the lecture [](mlp.md), we saw how the Multilayer Perceptron (a.k.a. Feedforward Neural Network, or Fully Connected Neural Network) generalizes linear models by stacking many affine transformations and placing nonlinear activation functions in between. Also, by the Universal Approximation Theorem, we saw that such a construction is enough to learn any function. But is an MLP always practical?

In this subsection, we will concentrate on working with images. Imagine that we have an image with 1000x1000 pixels and 3 RGB channels. If we take an MLP with one hidden layer of size 1000, this means that the weight matrix from input to layer 1 would have 3 billion parameters to map all 3M inputs to each of the 1k neurons in layer 1. This number is significantly large for most modern consumer hardware and thus such a network could not be easily trained or deployed.

MLPs are in a sense the most brute-force deep learning technique. By directly connecting all input entries to all next-layer neurons, we don't introduce any model bias, but this is (at least currently) just too hard for image data.

The core idea of Convolutional Neural Networks is to introduce weight sharing, i.e. different regions of the image are treated with the same weights.

## Convolution

```{figure} ../imgs/cnn/conv_eq.png
---
width: 400px
align: center
name: conv_eq
---
Continuous convolution equation (Source: [Intuitive Guide to Convolution](https://betterexplained.com/articles/intuitive-convolution/))
```

Applying a convolution filter (a.k.a. kernel) to a 2D image might look like this.

```{figure} ../imgs/cnn/conv_image.png
---
width: 300px
align: center
name: conv_image
---
Image convolution (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html))
```

### Filters

In image processing, specific (convolution) kernels have a well-understood meaning. Examples include:

- Edge detection
- Sharpening
- Mean blur
- Gaussian blur

```{figure} ../imgs/cnn/cnn_filters.png
---
width: 450px
align: center
name: cnn_filters
---
Examples of convolutional kernels (Source: [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing)))
```

The kernels is modern deep learnig lead to features like these:

```{figure} ../imgs/cnn/cnn_features.png
---
width: 500px
align: center
name: cnn_features
---
Intermediate feature maps (Image credit: Yann LeCun 2016, adapted from [Zeiler & Fergus 2013](https://arxiv.org/pdf/1311.2901.pdf))
```

## Dimensions of a Convolution

If you got excited about CNNs and open the PyTorch documentation of the corresponding `torch.nn.Conv2d` section, you will find the following notation:

- Input shape: $(N, C_{in}, H_{in}, W_{in})$
- Output shape: $(N, C_{out}, H_{out}, W_{out})$

Here, $N$ is the batch size, $C$ the number of channels, e.g. 3 for an RGB input image, $H$ is the height, and $W$ the width of an image. Using this notation we can compute the output height $H_{out}$ and output width $W_{out}$ of a CNN layer:

$$
H_{out}= \left\lfloor \frac{H_{in} + 2\cdot \text{padding}[0] - \text{dilation}[0] \cdot (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1 \right\rfloor,
$$ (cnn_height_out)

$$
W_{out}=\left\lfloor \frac{W_{in} + 2\cdot \text{padding}[1] - \text{dilation}[1] \cdot (\text{kernel_size}[1]-1) - 1}{\text{stride}[1]} + 1 \right\rfloor.
$$ (cnn_width_out)

Here, $\lfloor \cdot \rfloor$ denotes the floor operator. Let's look at what each of these new terms means.

### Padding

Applying a convolution directly to an image would result in an image of a smaller height and width. To counteract that, we pad the image height and width for example with zeros. With proper padding, one can stack hundreds of convolution layers without changing the width and height. The padding can be different along the width and height dimensions; we denote width and height padding with $\text{padding}[0]$ and $\text{padding}[1]$. The original convolution corresponds to $\text{padding}=0$.

```{figure} ../imgs/cnn/cnn_padding.png
---
width: 400px
align: center
name: cnn_padding
---
Convolution with zero padding (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html))
```

Another reason for padding is to use the corner pixels equally often as other pixels. The image below shows how often a pixel would be used by a convolutoin kernel of size 1x1, 2x2, and 3x3 without padding.

```{figure} ../imgs/cnn/cnn_without_padding.png
---
width: 500px
align: center
name: cnn_without_padding
---
Convolution without padding (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html))
```

Two commonly used terms regarding padding are the following:
- *valid* convolution: no padding
- *same* convolution: $H_{in}=H_{out}, \; W_{in}=W_{out}$.

### Stride

If we want to reduce the overlap between kernels and also reduce $W$ and $H$ of the outputs, we can introduce a $\text{stride}>1$ variable. $\text{stride}=1$ results in the original convolution. In the image below we see $\text{stride}=\text{Array}([2, 3])$.

```{figure} ../imgs/cnn/cnn_stride.png
---
width: 350px
align: center
name: cnn_stride
---
Convolution with stride (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html))
```

### Dilation

This is a more rare operation, which works well for detecting large-scale features. $\text{dilation}=1$ corresponds to the original convolution.

```{figure} ../imgs/cnn/dilation.gif
---
width: 200px
align: center
name: dilation
---
Convolution with dilation (Source: [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic))
```

## Pooling

You can think of a convolution filter as a *feature extraction* transformation similar to the basis expansion with general linear models. Here, the basis itself is learned via CNN layers.

If we are interested in image classification, we don't just want to transform the input features, but also *extract / select* the relevant information. This is done by pooling layers in between convolution layers. Pooling layers don't have learnable parameters and they inevitably reduce dimensionality. Typical examples are:

- max / min pooling
- mean pooling (averaging)

```{figure} ../imgs/cnn/cnn_pooling.png
---
width: 250px
align: center
name: cnn_pooling
---
Max pooling (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html))
```

### Channels

Each convolutional kernel operates on all $C_{in}$ input channels, resulting in a number of parameters per kernel $C_{in} \cdot \text{kernel_size}[0] \cdot \text{kernel_size}[1]$. Having $C_{out}$ number of kernels results in a number of parameters per convolutional layer given by

$$\# \text{params} = C_{in} \cdot C_{out} \cdot \text{kernel_size}[0] \cdot \text{kernel_size}[1]$$ (cnn_num_params)

The following is an example with two input channels, one output channel, and a 2x2 kernel size.

```{figure} ../imgs/cnn/cnn_multichannel.png
---
width: 450px
align: center
name: cnn_multichannel
---
Convolution of a 2-channel input with a 2x2 kernel (Source: {cite}`zhang2021`, [here](https://d2l.ai/chapter_convolutional-neural-networks/channels.html))
```

## Modern CNNs

A typical CNN would look something like the following.

```{figure} ../imgs/cnn/cnn_modern.png
---
width: 500px
align: center
name: cnn_modern
---
Modern CNN (Source: [cs231n, CNN lecture](http://cs231n.stanford.edu/slides/2021/lecture_5.pdf))
```

We now head to a historical overview of the trends since the beginning of the deep learning revolution with AlexNet in 2012.

### [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (2012)

Characteristics:
- rather larger filters with 11x11
- first big successes of ReLU
- 60M parameters

```{figure} ../imgs/cnn/cnn_alexnet.png
---
width: 600px
align: center
name: cnn_alexnet
---
AlexNet architecture (Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))
```

### [VGGNet](https://arxiv.org/abs/1409.1556) (2014)

Characteristics:
- much simpler structure
- 3x3 convolutions, stride 1, same convolutions
- 2x2 max pooling
- deeper
- 138M parameters

```{figure} ../imgs/cnn/cnn_vggnet.png
---
width: 600px
align: center
name: cnn_vggnet
---
VGGNet architecture (Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))
```

### [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (2016)

Charasteristics:
- allows for very deep networks by introducing skip connections
- this mitigates the vanishing and exploding gradients problem
- ResNet-152 (with 152 layers) has 60M parameters

```{figure} ../imgs/cnn/cnn_resnet.png
---
width: 700px
align: center
name: cnn_resnet
---
ResNet architecture (Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))
```

### [U-Net](https://arxiv.org/abs/1505.04597) (2015)

Characteristics:
- for image segmentation
- skip connections

```{figure} ../imgs/cnn/cnn_unet.png
---
width: 600px
align: center
name: cnn_unet
---
U-Net architecture (Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))
```

### Advanced Topics: [ConvNext](https://arxiv.org/abs/2201.03545) (2022) and [ConvNextv2](https://arxiv.org/abs/2301.00808) (2023)

Characteristics:
- since the [Vision Transformer](https://arxiv.org/abs/2010.11929) many people started believing that the inductive bias of translational invariance encoded in a convolution is too restrictive for images classification. However, the release of the [ConvNext](https://arxiv.org/abs/2201.03545) model ("A ConvNet for the 2020s", Liu et al. 2022) points in the direction that many innovations have been made on improving transformers, e.g. the GELU activations, and if we simply apply some of them to CNNs, we also end up with state-of-the-art results.

```{figure} ../imgs/cnn/cnn_convnext.png
---
width: 450px
align: center
name: cnn_convnext
---
ConvNext ablations (Source: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545))
```

The successor paper of ConvNext -> ConvNextv2 came out one week before this lecture in WS22/23 :)

## Further References

- {cite}`goodfellow2016`, Chapter 9
- {cite}`zhang2021`, Chapters "Convolutional Neural Networks" and "Modern Convolutional Neural Networks"
