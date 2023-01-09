# Convolutional Neural Network

## Limitations of MLP

In the previous subsection, we saw how the Multilayer Perceptron (a.k.a. Feedforward Neural Network, or Fully Connected Neural Network) generalizes linear models by stacking many linear models and placing nonlinear activation functions in between. Also, by the Universal Approximation Theorem, we saw that such a construction is enough to learn any function. But is an MLP always practical?
1AJ3d98
In this subsection, we will concentrate on working with images. Imagine that we have an image with 1000x1000 pixels and 3 RGB channels. If we take an MLP with one hidden layer of size 1000, this means that the weight matrix from input to layer 1 would have 3 billion parameters to map all 3M inputs to each of the 1k neurons in layer 1. This number is too large for most modern consumer hardware and thus such a network could not be easily trained or deployed.

MLPs are in a sense the most brute-force deep learning technique. By directly connecting all inputs to all next-layer neurons, we don't introduce any bias and this is at least currently just too hard for image data.

The core idea of Convolutional Neural Networks is to introduce weight sharing, i.e. different regions of the image are treated with the same weights.

## Convolutions

<div style="text-align:center">
    <img src="https://i.imgur.com/1AJ3d98.png" alt="drawing" width="500"/>
</div>

(Source: [Intuitive Guide to Convolution](https://betterexplained.com/articles/intuitive-convolution/))

Applying a convolution filter (a.k.a. kernel) to a 2D image looks might look like this.

<div style="text-align:center">
    <img src="https://i.imgur.com/ygH2Go6.png" alt="drawing" width="350"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html))

### Filters

In image processing, specific (convolution) kernels have a well-understood meaning. Examples include:

- Edge detection
- Sharpening
- Mean blur
- Gaussian blur

<div style="text-align:center">
    <img src="https://i.imgur.com/PKCBAVh.png" alt="drawing" width="500"/>
</div>

(Source: [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing))

The kernels is modern deep learnig look more like this:


<div style="text-align:center">
    <img src="https://i.imgur.com/xvljdrQ.png" alt="drawing" width="600"/>
</div>

(Image credit: Yann LeCun 2016, adapted from [Zeiler & Fergus 2013](https://arxiv.org/pdf/1311.2901.pdf)   


## Dimensions of a Convolution

If you got excited about CNNs and open the documentation to one of the most popular ML libraries PyTorch, in the corresponding `torch.nn.Conv2d` section you will find the following equations.

Input: $(C_{in}, H_{in}, W_{in})$
Output: $(C_{out}, H_{out}, W_{out})$

Here, $C$ is the number of channels, e.g. 3 for an RGB input image, $H$ is the height, and $W$ is the width of an image.

$$H_{out}= \left\lfloor \frac{H_{in} + 2\cdot \text{padding}[0] - \text{dilation}[0] \cdot (\text{kernel\_size}[0]-1) - 1}{\text{stride}[0]} + 1 \right\rfloor$$

$$W_{out}=\left\lfloor \frac{W_{in} + 2\cdot \text{padding}[1] - \text{dilation}[1] \cdot (\text{kernel\_size}[1]-1) - 1}{\text{stride}[1]} + 1 \right\rfloor$$

With $\lfloor \cdot \rfloor$ we denote the floor operator. Let's look at what each of these new terms means.

### Padding

Applying a convolution directly to an image would result in an image of a smaller height and width. To counteract that, we pad the image height and width for example with zeros. With proper padding, one can stack hundreds of convolution layers without changing the width and height. The padding can be different along the width and height dimensions; we denote width and height padding with $\text{padding}[0]$ and $\text{padding}[1]$. $\text{padding}=0$ is the original convolution.

<div style="text-align:center">
    <img src="https://i.imgur.com/swEdDMq.png" alt="drawing" width="500"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html))

Another reason for padding is to use the corner pixels equally often as other pixels. The image below shows how often a pixel would be used by a convolutoin kernel of size 1x1, 2x2, and 3x3 without padding.

<div style="text-align:center">
    <img src="https://i.imgur.com/T0NkS8w.png" alt="drawing" width="500"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html))

Two commonly used terms regarding padding are the following:
- *valid* convolution: no padding
- *same* convolution: $H_{in}=H_{out}, \; W_{in}=W_{out}$.

### Stride

If we want to reduce the overlap between kernels and also reduce $W$ and $H$ of the outputs, we can introduce a $\text{stride}>1$ variable. $\text{stride}=1$ results in the original convolution. In the image below we see $\text{stride}=\text{Array}([2, 3])$.

<div style="text-align:center">
    <img src="https://i.imgur.com/Ays8u2j.png" alt="drawing" width="400"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html))


### Dilation

This is a more exotic operation, which works well for detecting large-scale features. $\text{dilation}=1$ corresponds to the original convolution.

![Optimization](../imgs/dilation.gif)

(Source: [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic))


## Pooling

You can think of a convolution filter as a feature extraction transformation similar to the basis expansion with general linear models. Here, the basis itself is learned via CNN layers. 

If we are interested in image classification, we don't just want to transform the input features, but also extract / select the relevant information. This is done by pooling layers in between convolution layers. Pooling layers don't have learnable parameters and the inevitable reduce dimensionality. Typical examples are:

- max / min pooling
- mean pooling (averaging)


<div style="text-align:center">
    <img src="https://i.imgur.com/c1s6T2F.png" alt="drawing" width="300"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html))


### Channels

Each convolution kernel operates on all $C_{in}$ input channels, resulting in a number of parameters per kernel $C_{in} \cdot \text{kernel\_size}[0] \cdot \text{kernel\_size}[1]$. Having $C_{out}$ number of kernels results in a number of parameters per convolution layer of 

$C_{in} \cdot C_{out} \cdot \text{kernel\_size}[0] \cdot \text{kernel\_size}[1]$

<div style="text-align:center">
    <img src="https://i.imgur.com/JMVifDx.png" alt="drawing" width="550"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_convolutional-neural-networks/channels.html))


## Modern CNNs

A typical CNN would look something like the following.


<div style="text-align:center">
    <img src="https://i.imgur.com/YQLkWG9.png" alt="drawing" width="550"/>
</div>

(Source: [cs231n, CNN lecture](http://cs231n.stanford.edu/slides/2021/lecture_5.pdf))

We now head to a historical overview of the recent trends since the beginning of the deep learning revolution with AlexNet in 2012.

### [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (2012)

Characteristics:
- rather larger filters with 11x11
- first big successes of ReLU
- 60M parameters

<div style="text-align:center">
    <img src="https://i.imgur.com/ylA5l5O.png" alt="drawing" width="600"/>
</div>

(Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))

### [VGGNet](https://arxiv.org/abs/1409.1556) (2014)

Characteristics:
- much simpler structure
- 3x3 convolutions, stride 1, same convolutions
- 2x2 max pooling
- deeper
- 138M parameters

<div style="text-align:center">
    <img src="https://i.imgur.com/Pxekgh6.png" alt="drawing" width="600"/>
</div>

(Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))

### [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (2016)

Charasteristics:
- allows for very deep networks by introducing skip connections
- this mitigates the vanishing and exploiding gradients problem
- ResNet-152 (with 152 layers) has 60M parameters

<div style="text-align:center">
    <img src="https://i.imgur.com/6gwMcYz.png" alt="drawing" width="700"/>
</div>

(Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))


### [U-Net](https://arxiv.org/abs/1505.04597) (2015)

Characteristics:
- for image segmentation
- skip connections

<div style="text-align:center">
    <img src="https://i.imgur.com/kk2asgp.png" alt="drawing" width="600"/>
</div>

(Source: [I2DL, TUM](https://niessner.github.io/I2DL/slides/10.CNN-2.pdf))


### [ConvNext](https://arxiv.org/abs/2201.03545) (2022) and [ConvNextv2](https://arxiv.org/pdf/2301.00808.pdf) (2023)

Characteristics:
- since the [Vision Transformer](https://arxiv.org/abs/2010.11929) many people started believing that the inductive bias of translational invariance encoded in a convolution is too restrictive for images classification. However, the release of the [ConvNext](https://arxiv.org/abs/2201.03545) model ("A ConvNet for the 2020s", Liu et al. 2022) points in the direction that many innovations have been made on improving transformers, e.g. the GELU activations, and if we simply apply some of them to CNNs, we also end up with state-of-the-art results.

<div style="text-align:center">
    <img src="https://i.imgur.com/CQ0hNVZ.png" alt="drawing" width="500"/>
</div>

(Source: [ConvNext](https://arxiv.org/abs/2201.03545))

The successor paper of ConvNext -> ConvNextv2 just came out one week ago!

## Further References

- [Deep Learning](https://www.deeplearningbook.org/), Chapters 6; Goodgellow, Bengio, Courville; 2016
- [d2l](https://d2l.ai/index.html), Chapter 6 and 7; Zhang et al.; 2022