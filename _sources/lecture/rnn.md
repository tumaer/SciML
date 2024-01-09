# Recurrent Models

While CNNs, and MLPs are excellent neural network architectures for **spatial** relations, they yet struggle with the modeling of **temporal** relations which they are incapable of modeling in their default configuration. The figure below gives an overview of different task formulations.

```{figure} ../imgs/rnn_sequences.jpeg
---
width: 600px
align: center
name: rnn_sequences
---
Types of sequences (Source: [karpathy.github.io](https://karpathy.github.io/2015/05/21/rnn-effectiveness/))
```

For such tasks there exist a number of specialized architectures most notably:

1. [Recurrent Neural Networks](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
2. [Long Short-term Memory (LSTM) networks](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)
3. [Transformers](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

While Transformers have become the dominant architecture in machine learning these days, their roots lie in the development of RNNs from whom we will begin to build up the content of this section to introduce the architectures in order, show their similarities, as well as special properties and where you most appropriately deploy them.

## Recurrent Neural Networks (RNNs)

Where we before mapped from an input space of e.g. images, Recurrent Neural Networks (RNNs) map from an input space of sequences, to an output space of sequences. Where their core property is the **stateful** prediction, i.e. if we seek to predict an output $y$, then $y$ depends not only only on the input $x$ but also on the **hidden state of the system** $h$. The hidden state of the neural network is updated as time progresses during the processing of the sequence. There are a number of usecases for such model such as:

* Sequence Generation
* Sequence Classification
* Sequence Translation

We will be focussing on the three in the very same order.

### Sequence Generation

Sequence generation can mathematically be summarized as

$$
f_{\theta}: \mathbb{R}^{D} \longrightarrow \mathbb{R}^{N_{\infty}C}
$$ (rnn_vec2seq)

with an input vector of size $D$, and an output sequence of $N_{\infty}$ vectors each with size $C$. As we are essentially mapping a vector to a sequence, these models are also called **vec2seq** models. The output sequence is generated one token at a time, where we sample at each step from the current hidden state $h_{t}$ of our neural network, which is subsequently fed back into the model to update the hidden state to the new state $h_{t+1}$.

```{figure} ../imgs/rnn_vec2seq.png
---
width: 250px
align: center
name: rnn_vec2seq
---
Vector to sequence task (Source: {cite}`murphy2022`, Chapter 15)
```

To summarize, a vec2seq model is a probabilistic model of the form $p(y_{1:T}|x)$. If we now break this probabilistic model down into its actual mechanics, then we end up with the following conditional **generative model**

$$
p(y_{1:T}|x) = \sum_{h_{1:T}} p(y_{1:T}, h_{1:T} | x) = \sum_{h_{1:T}} \prod^{T}_{t=1} p(y_{t}|h_{t})p(h_{t}|h_{t-1}, y_{t-1}, x).
$$ (vec2seq_model)

Just like a Runge-Kutta scheme, this model requires the seeding with an initial hidden state distribution. This distribution has to be predetermined and is most often deterministic. The computation of the hidden state is then presumed to be

$$
p(h_{t}|h_{t-1}, y_{t-1}, x) = \mathbb{I}(h_{t}=f(h_{t-1}, y_{t-1}, x))
$$ (rnn_prob_h)

for the deterministic function $f$. A typical choice constitutes

$$
h_{t} = \varphi(W_{xh}x_{t} + W_{hh}h_{t-1} + b_{h}),
$$ (rnn_simplest_h)

with $W_{hh}$ the hidden-to-hidden weights, and $W_{xh}$ the input-to-hidden weights. The output distribution is then either given by

$$
p(y_{t}|h_{t}) = \text{Cat}(y_{t}| \text{softmax}(W_{hy}h_{t} + b_{y})),
$$ (rnn_prob_h2y)

where "Cat" is the categorical distribution in the case of a discrete output from a predefined set, or by something like

$$
p(y_{t}|h_{t}) = \mathcal{N}(y_{t}|W_{hy} h_{t} + b_{y}, \sigma^{2}{\bf{I}})
$$

for real-valued outputs. Now if we seek to express this in code, then our model looks something like this:

```python
def rnn(inputs, state, params):
    # Referring to Karpathy's drawing above, we present the right-most case many-to-many
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

If you remember one thing about this section:

* The key to RNNs is their unbounded memory, which allows them to make more stable predictions, and also remember further back in time.
* The stochasticity in the model comes from the noise in the output model.

> To forecast spatio-temporal data, one has to combine RNNs with CNNs. The classical form of this is the [convolutional LSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf).

```{figure} ../imgs/rnn_conv_rnn.png
---
width: 600px
align: center
name: rnn_conv_rnn
---
CNN-RNN (Source: {cite}`murphy2022`, Chapter 15)
```

### Sequence Classification

If we now presume to have a fixed-length output vector, but with a variable length sequence as input, then we mathematically seek to learn

$$
f_{\theta}: \mathbb{R}^{TD} \longrightarrow \mathbb{R}^{C}
$$ (rnn_seq2vec)

this is called a **seq2vec** model. Here we presume the output $y$ to be a class label.

```{figure} ../imgs/rnn_seq2vec.png
---
width: 300px
align: center
name: rnn_seq2vec
---
Sequence to vector task (Source: {cite}`murphy2022`, Chapter 15)
```

In its simplest form we can just use the final state of the RNN as the input to the classifier

$$
p(y|x_{1:T}) = \text{Cat}(y| \text{softmax}(Wh_{T}))
$$ (seq2vec_model)

While this simple form can already produce good results,  the RNN principle can be extended further by allowing **information to flow in both directions**, i.e. we allow the hidden states to depend on past and future contexts. For this we have to use two basic RNN building blocks, to then assemble them into a **bidirectional RNN**.

```{figure} ../imgs/rnn_bidir.png
---
width: 250px
align: center
name: rnn_bidir
---
Bi-directional RNN for seq2vec tasks (Source: {cite}`murphy2022`, Chapter 15)
```

This bi-directional model is then defined as

$$
\begin{align}
    h_{t}^{\rightarrow} &= \varphi(W_{xh}^{\rightarrow}x_{t} + W_{hh}^{\rightarrow}h_{t-1}^{\rightarrow} + b_{h}^{\rightarrow}) \\
    h_{t}^{\leftarrow} &= \varphi(W_{xh}^{\leftarrow} x_{t} + W_{hh}^{\leftarrow} h_{t+1}^{\leftarrow} + b_{h}^{\leftarrow}),
\end{align}
$$ (rnn_bidir_h_eq)

where the hidden state then transforms into a vector of forward-, and reverse-time hidden state

$$
h = [ h_{t}^{\rightarrow}, h_{t}^{\leftarrow} ].
$$ (rnn_bidir_h)

One has to then average pool over these states to arrive at the predictive model

$$
\begin{align}
    p(y|x_{1:T}) &= \text{Cat}(y| W \hspace{2pt} \text{softmax}(\bar{h})) \\
    \bar{h} &= \frac{1}{T} \sum_{t=1}^{T} h_{t}
\end{align}
$$ (rnn_bidir_pooling)

### Sequence Translation

In sequence translation we have a variable length sequence as an input and a variable length sequence as an output. This can mathematically be expressed as

$$
f_{\theta}: \mathbb{R}^{TD} \rightarrow \mathbb{R}^{T'C}.
$$ (seq2seq_model)

For ease of notation, this has to be broken down into two subcases:

1. $T'=T$ i.e. we have the same length of input- and output-sequences
2. $T' \neq T$, i.e. we have different lengths between the input- and the output-sequence

```{figure} ../imgs/rnn_seq2seq.png
---
width: 400px
align: center
name: rnn_seq2seq
---
Encoder-Decoder RNN for sequence to sequence task (Source: {cite}`murphy2022`, Chapter 15)
```

#### Aligned Sequences

We begin by examining the case for $T'=T$, i.e. with the same length of input-, and output-sequences. In this case  we have to predict one label per location, and can hence modify our existing RNN for this task

$$
p(y_{1:T}| x_{1:T}) = \sum_{h_{1:T}} \prod^{T}_{t=1} p(y_{t}|h_{t}) \mathbb{I}(h_{t} = f(h_{t-1}, x_{t}))
$$ (rnn_prob_seq2seq)

Once again, results can be improved by allowing for bi-directional information flow with a bidirectional RNN which can be constructed as shown before, or by **stacking multiple layers on top of each other** and creating a **deep RNN**.

```{figure} ../imgs/rnn_deep.png
---
width: 400px
align: center
name: rnn_deep
---
Deep RNN (Source: {cite}`murphy2022`, Chapter 15)
```

In the case of stacking layers on top of each other to create deeper networks, we have hidden layers lying on top of hidden layers. The individual layers are then computed with

$$
h_{t}^{l} = \varphi_{l}(W^{l}_{xh} h^{l-1}_{t} + W^{l}_{hh} h^{l}_{t-1} + b_{h}^{l})
$$ (rnn_layer)

and the output are computed from the final layer

$$
o_{t} = W_{ho} h_{t}^{L} + b_{o}.
$$ (rnn_deep_output)

#### Unaligned Sequences

In the unaligned case we have to learn a mapping from the input-sequence to the output-sequence, where we first have to encode the input sequence into a context vector

$$
c = f_{e}(x_{1:T}),
$$ (rnn_context_vector)

using the last state of an RNN or pooping over all states. We then generate the output sequence using a decoder RNN, which leads to the so called **encoder-decoder architecture**. There exist a plethora of tokenizers to construct the context vectors, and a plethora of decoding approaches such as the greedy decoding shown below.

```{figure} ../imgs/rnn_seq2seq_translate.png
---
width: 800px
align: center
name: rnn_seq2seq_translate
---
Sequence to sequence translation of English to French using greedy decoding (Source: {cite}`murphy2022`, Chapter 15)
```

This *encoder-decoder* architecture dominates general machine learning, as well as scientific machine learning. Examining the use-cases you have seen up to now:

* U-Net
* Convolutional LSTM, i.e. encoding with CNNs, propagating in time with the LSTM, and then decoding with CNNs again
* Sequence Translation as just now

And an unending list of applications which you have seen in practice but have not seen in the course yet

* Transformer models
  * GPT
  * BERT
  * ChatGPT
* Diffusion models for image generation
* ...

> But if RNNs can already do so much, why are Transformers then dominating machine learning research these days and not RNNs?

Training RNNs is far from trivial with a well-known problem being **exploding gradients**, and **vanishing gradients**. In both cases the activations of the RNN explode or decay as we go forward in time as we multiply with the weight matrix $W_{hh}$ at each time step. The same can happen as we go backwards in time, as we repeatedly multiply the Jacobians and unless the spectrum of the Hessian is 1, this will result in exploding or vanishing gradients. A way to tackle this is via **control of the spectral radius** where the optimization problem gets converted into a convex optimization problem, which is then called an **echo state network**. Which is in literature often used under the umbrella term of **reservoir computing**.


## Long Short-term Memory (LSTM)

A way to avoid the problem of exploding and vanishing gradients, beyond Gated Recurrent Units (GRU) which we omit in this course, is the long short term memory (LSTM) model of Schmidhuber and Hochreiter, back in the day at TUM. In the LSTM the hidden state $h$ is augmented with a **memory cell $c$**. This cell is then controlled with 3 gates

* Output gate $O_{t}$
* Input gate $I_{t}$
* Forget gate $F_{t}$

where the forget gate determines when the memory cell is to be reset. The individual cells are then computed as

$$
\begin{align}
    O_{t} &= \sigma(X_{t}W_{xo} + H_{t-1}W_{ho} + b_{o}) \\
    I_{t} &= \sigma(X_{t} W_{xi} + H_{t-1} W_{hi} + b_{i}) \\
    F_{t} &= \sigma(X_{t}W_{xf} H_{t-1}W_{hf} + b_{f}), 
\end{align}
$$ (lstm_gates)

from which the cell state can then be computed as

$$
\tilde{C}_{t}= \tanh(X_{t} W_{xc} + H_{t-1}W_{hc} + b_{c})
$$ (lstm_cell_state_interm)

with the actual update then given by either the candidate cell, if the input gate permits it, or the old cell, if the not-forget gate is on

$$
C_{t} = F_{t} \odot C_{t-1} + I_{t} \odot \tilde{C}_{t}.
$$ (lstm_cell_state)

The hidden state is then computed as a transformed version of the memory cell if the output gate is on

$$
H_{t} = O_{t} \odot \tanh(C_{t})
$$ (lstm_hidden)

Visually this then looks like the following:

```{figure} ../imgs/lstm_block.png
---
width: 450px
align: center
name: lstm_block
---
LSTM block (Source: {cite}`murphy2022`, Chapter 15)
```

This split results in the following properties:

* $H_{t}$ acts as a short-term memory
* $C_{t}$ acts as a long-term memory

In practice this then takes the following form in code:

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

There exists a great many variations on this initial architecture, but the core LSTMs' architecture as well as performance have prevailed over time so far. A different approach to sequence generation is the use of causal convolutions with 1-dimensional CNNs. While this approach has shown promise in quite a few practical applications, we view it as not relevant to the exam.

## Advanced Topics: 1-Dimensional CNNs

While RNNs have very strong temporal prediction abilities with their memory, as well as stateful computation, 1-D CNNs can constitute a viable alternative as they don't have to carry along the long term hidden state, as well as being easier to train as they do not suffer from exploding or vanishing gradients.

### Sequence Classification

Recalling, for sequence classification we consider the seq2vec case, in which we have a mapping of the form

$$
f_{\theta}: \mathbb{R}^{TD} \rightarrow \mathbb{R}^{C}.
$$ (rnn_seq2vec_duplicate)

A 1-D convolution applied to an input sequence of length $T$, and $D$ features per input then takes the form

```{figure} ../imgs/rnn_textcnn.png
---
width: 500px
align: center
name: rnn_textcnn
---
TextCNN architecture (Source: {cite}`murphy2022`, Chapter 15)
```

With $D>1$ input channels of each input sequence, each channel is then convolved separately and the results are then added up with each channel having its own separate 1-D kernel s.t. (recalling from the CNN lecture)

$$
z_{i} = \sum_{d} x^{\top}_{i-k:i+k,d} w_{d},
$$ (rnn_textcnn)

with $k$ being the size of the receptive field, and $w_{d}$ the filter for the input channel $d$. Which produces a 1-D input vector $z \in \mathbb{R}^{T}$ (ignoring boundaries), i.e. for each output channel $c$ we then get

$$
z_{ic} = \sum_{d} x^{\top}_{i-k:i+k, d} w_{d, c}.
$$ (rnn_textcnn_channelwise)

To then reduce this to a fixed size vector $z \in \mathbb{R}^{C}$, we have to use max-pooling over time s.t.

$$
z_{c} = \max_{i} z_{ic},
$$ (rnn_textcnn_pooled)

which is then passed into a softmax layer. What this construction permits is that by choosing kernels of different widths we can essentially use a library of different filters to capture patterns of different frequencies (length scales). In code this then looks the following:

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size) # not being trained
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # no weight and can hence share the instance
        self.relu = nn.ReLU()
        # Creating the one-dimensional convolutional layers with different kernel widths
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenation of the two embedding layers
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Channel dimension of the 1-D conv layer is transformed
        embeddings = embeddings.permute(0, 2, 1)
        # Flattening to remove overhanging dimension, and concatenate on the channel dimension
        # For each one-dimensional convolutional layer, after max-over-time
        encoding = torch.cat(
            [torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs], dim=1
        )
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

### Sequence Generation

To use CNNs generatively, we need to introduce **causality** into the model, this results in the model definition of

$$
p(y) = \prod_{t=1}^{T} p(y_{t}|y_{1:t-1}) = \prod_{t=1}^{T} \text{Cat}(y_{t}| \text{softmax}(\varphi(\sum_{\tau = 1}^{t-k}w^{\top}y_{\tau:\tau+k}))),
$$ (rnn_causalcnn)

where we have the convolutional filter $w$ and a nonlinearity $\varphi$. This results in a masking out of future inputs, such that $y_{t}$ can only depend on past information, and no future information. This is called a **causal convolution**. One poster-child example of this approach is the [WaveNet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio) architecture, which takes in text as input sequences to generate raw audio such as speech. The WaveNet architecture utilizes dilated convolutions in which the dilation increases with powers of 2 with each successive layer.

```{figure} ../imgs/rnn_causalcnn.png
---
width: 600px
align: center
name: rnn_causalcnn
---
WaveNet architecture (Source: {cite}`murphy2022`, Chapter 15)
```

This then takes following form in [code](https://github.com/antecessor/Wavenet/blob/master/Wavenet.py).

## Flagship Applications

With so many flagship models of AI these days relying on sequence models, we have compiled a list of a very few of them below for you to play around with. While attention was beyond the scope of this lecture, you can have a look at Lilian Weng's blog post on attention below to dive into it.

**Large Language Models (LLMs)**
* [Cohere AI Blog Posts](https://txt.cohere.ai/generative-ai-part-1/)
* [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)
    * [GPT-3](https://openai.com/blog/gpt-3-apps/) - where ChatGPT started
* [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) - Google's old open source LLM
* [Llama 2](https://ai.meta.com/llama/) - Meta's open source LLMs
* [Mistral/Mistral](https://mistral.ai/product/) - Mistral AI's open source LLMs

**Others**
* [OpenAI's Dall-E 2](https://openai.com/dall-e-2/) - image generation
* [Stable Diffusion](https://stablediffusionweb.com/#ai-image-generator) - open source image generation
* [Whisper](https://github.com/openai/whisper) - open source audio to text model


## Further Reading

* {cite}`murphy2022`, Chapter 15 - main reference for this lecture
* [RNN blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [LSTM blog post by Chris Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs)
* [Lilian Weng's blog post on Transformers](https://lilianweng.github.io/posts/2018-06-24-attention/)
