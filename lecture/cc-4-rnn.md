# Recurrent Models

While CNNs, and MLPs are excellent neural networkc architectures for **spatial** relations, they yet struggle with the modeling of **temporal** relations which they are incapable of modeling. For this task there exist a number of specialized architectures most notably:

1. [Recurrent Neural Networks](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
2. [Long Short-term Memory (LSTM) networks](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)
3. [Transformers](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

While Transformers have become the dominant architecture in machine learning these days, their roots lie in the development of RNNs from whom we will begin to build up the content of this section to introduce the architectures in order, show their similarities, as well as special properties and where you most appropriately deploy them.

## Recurrent Neural Networks (RNN)

Where we before mapped from an input space of e.g. images, Recurrent Neural Networks (RNNs) map from an input space of sequences, to an output space of sequences. Where their core property is the **stateful** prediction, i.e. if we seek to predict an output $y$, then $y$ depends not only only on the input $x$ but also on the **hidden state of the system** $h$. The hidden state of the neural network is updated as time progresses during the processing of the sequence. There are a number of usecases for such model such as:

* Sequence Generation
* Sequence Classification
* Sequence Translation

we will be focussing on sequence generation, and sequence translation in this lecture.

### Sequence Generation

Sequence generation can mathematically be summarized as

$$
f_{\theta}: \mathbb{R}^{D} \longrightarrow \mathbb{R}^{N_{\infty}C}
$$

with an input vector of size $D$, and an output sequence of $N_{\infty}$ vectors each with size $C$. As we are essentially mapping a vector to a sequence, these models are also called **vec2seq** models. The output sequence is generated one token at a time, where we sample at each step from the current hidden state $h_{t}$ of our neural network, which is subsequently fed back into the model to update the hidden state to the new state $h_{t+1}$.

<center>
    <img src = "https://i.imgur.com/ji2oQ7V.png" width = "350">
</center>

To summarize, a vec2seq model is a probabilistic model of the form

$$
p(y_{1:T}|x)
$$

if we now break this probabilistic model down into its actual mechanics then we end up with the following conditional **generative model**

$$
p(y_{1:T}|x) = \sum_{h_{1:T}} p(y_{1:T}, h_{1:T} | x) = \sum_{h_{1:T}} \prod^{T}_{t=1} p(y_{t}|h_{t})p(h_{t}|h_{t-1}, y_{t-1}, x)
$$

Just like a Runge-Kutta scheme, this requires this model requires the seeding with an initial hidden state distribution. This distribution has to be predetermined and is most often deterministic. The computation of the hidden state is then presumed to be 

$$
p(h_{t}|h_{t-1}, y_{t-1}, x) = \mathbb{I}(h_{t}=f(h_{t-1}, y_{t-1}, x))
$$

for the deterministic function $f$. A typical choice constitutes

$$
h_{t} = \varphi(W_{xh}[x;y_{t-1}] + W_{hh}h_{t-1} + b_{h})
$$

with $W_{hh}$ the hidden-to-hidden weights, and $W_{xh}$ the input-to-hidden weights. The output distribution is then either given by

$$
p(y_{t}|h_{t}) = \text{Cat}(y_{t}| \text{softmax}(W_{hy}h_{t} + b_{y}))
$$

where "Cat" is the concatenation of the outputs, or by

$$
p(y_{t}|h_{t}) = \mathcal{N}(y_{t}|W_{hy} h_{t} + b_{y}, \sigma^{2}{\bf{I}})
$$

for real-valued outputs. Now if we seek to express this in code, then our model looks something like this

```python
def rnn(inputs, state, params):
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

for the usual output. If you remember one thing about this section:

* The key to RNNs is their unbounded memory, which allows them to make more stable predictions, and also remember further back in time.
* The stochasticity in the model comes from the noise in the output model.

> To forecast spatio-temporal data, one has to combine RNNs with CNNs. The classical form of this is the [convolutional LSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf).

<center>
    <img src = "https://i.imgur.com/SMhdrjt.png" width = "400">
</center>

### Sequence Classification

If we now presume to have a fixed-length output vector, but with a variable length sequence as input, then we mathematically seek to learn

$$
f_{\theta}: \mathbb{R}^{TD} \longrightarrow \mathbb{R}^{C}
$$

this is called a **seq2vec** model. Here we presume the output $y$ to be a class label.

<center>
    <img src="https://i.imgur.com/3OPqG9c.png" width="400">
</center>

In its simplest form we can just use the final state of the RNN as the input to the classifier

$$
p(y|x_{1:T}) = \text{Cat}(y| \text{softmax}(Wh_{T}))
$$

While this simple form can already produce good results,  the RNN principle can be extended further by allowing **information to flow in both directions**, i.e. we allow the hidden states to depend on past and future contexts. For this we have to use two basic RNN building blocks, to then assemble them into a **bidirectional RNN**.

<center>
    <img src="https://i.imgur.com/yquh339.png" width="400">
</center>

the model is then defined as

$$
\begin{align}
    h_{t}^{\rightarrow} &= \varphi(W_{xh}^{\rightarrow} + W_{hh}^{\rightarrow}h_{t-1}^{\rightarrow} + b_{h}^{\rightarrow}) \\
    h_{t}^{\leftarrow} &= \varphi(W_{xh}^{\leftarrow} x_{t} + W_{hh}^{\leftarrow} h_{t+1}^{\leftarrow} + b_{h}^{\leftarrow})
\end{align}
$$

where the hidden state then transforms into a vector of forward-, and reverse-time hidden state

$$
h = [ h_{t}^{\rightarrow}, h_{t}^{\leftarrow} ]
$$

one has to then average pool over these states to arrive at the predictive model

$$
\begin{align}
    p(y|x_{1:T}) &= \text{Cat}(y| W \hspace{2pt} \text{softmax}(\bar{h})) \\
    \bar{h} &= \frac{1}{T} \sum_{t=1}^{T} h_{t}
\end{align}
$$

## Sequence Translation

In sequence translation we have a variable length sequence as an input. and a variable length sequence as an output. This can mathematically be expressed as

$$
f_{\theta}: \mathbb{R}^{TD} \rightarrow \mathbb{R}^{T'C}
$$

for ease of notation, this has to be broken down into two subcases:

1. $T'=T$ i.e. we have the same length of input- and output-sequences
2. $T' \neq T$, i.e. we have different lengths between the input- and the output-sequence

<center>
    <img src="https://i.imgur.com/yUbVJRN.png" width="400">
</center>

### Aligned Sequences

We begin by examining the case for $T'=T$, i.e. with the same length of input-, and output-sequences. In this case  we have to predict one label per location, and can hence modify our existing RNN for this task

$$
p(y_{1:T}| x_{1:T}) = \sum_{h_{1:T}} \prod^{T}_{t=1} p(y_{t}|h_{t}) \mathbb{I}(h_{t} = f(h_{t-1}, x_{t}))
$$

once again, results can be improved by allowing for bi-directional information flow with a bidirectional RNN which can be constructed as shown before, or by **stacking multiple layers on top of each other** and creating a **deep RNN**.

<center>
    <img src="https://i.imgur.com/jvWmqMo.png" width="400">
</center>

in the case of stacking layers on top of each other to create deeper networks, we have hidden layers lying on top of hidden layers. The individual layers are then computed with

$$
h_{t}^{l} = \varphi_{l}(W^{l}_{xh} h^{l-1}_{t} + W^{l}_{hh} h^{l}_{t-1} + b_{h}^{l})
$$

and the output being computed from the final layer

$$
o_{t} = W_{ho} h_{t}^{L} + b_{o}
$$

### Unaligned Sequences

In the unaligned case we have to learn a mapping from the input-sequence to the output-sequence, where we first have to encode the input sequence into context vectors

$$
c = f_{e}(x_{1:T})
$$

using the last state of an RNN, before then generating the output sequence using an RNN where there exist a plethora of tokenizers to construct the context vectors, and a plethora of decoding approaches such as the greedy decoding shown below.

<center>
    <img src="https://i.imgur.com/oNvsDLN.png" width="400">
    <img src="https://i.imgur.com/Puui40V.png" width="400">
</center>

this is what is called a **encoder-decoder architecture** which dominates general machine learning, as well as scientific machine learning. Examining the use-cases you have seen up to now:

* U-Net
* Convolutional LSTM, i.e. encoding with CNNs, propagating in time with the LSTM, and then decoding with CNNs again
* Sequence Translationa as just now

and an unending list of applications which you have seen in practice but have not seen in the course yet

* Transformer models
  * GPT
  * BERT
  * ChatGPT
* Stable Diffusion
* ...

> But if RNNs can already do so much, why are Transformers then dominating machine learning research these days and not RNNs?

Training RNNs is far from trivial with a well-known problem being **exploding gradients**, and **vanishing gradients**. In both cases the activations of the RNN explode or decay as we go forward in time as we multiply with the weight matrix $W_{hh}$ at each time step. The same can happen as we go backwards in time, as we repeatedly multiply the Jacobians and unless the spectrum of the Hessian is 1, this will result in exploding or vanishing gradients. A way to tackle this is via **control of the spectral radius** where the optimization problem gets converted into a convex optimization problem, which is then called an **echo state network**. Which is in literature often used under the umbrella term of **reservoir computing**.

## Long Short-term Memory (LSTM)

A way to avoid this problem, beyond Gated Recurrent Units (GRU) which we omit in this course, is the long short term memory (LSTM) model of Schmidhuber. In the LSTM the hidden state $h$ is augmented with a **memory cell c**. This cell is then controlled with 3 gates

* Output gate $O_{t}$
* Input gate $I_{t}$
* Forget gate $F_{t}$

where the forget gate determines when the memory cell is to be reset. The individual cells are then computed as

$$
\begin{align}
    O_{t} &= \sigma(X_{t}W_{xo} + H_{t-1}W_{ho} + b_{o}) \\
    I_{t} &= \sigma(X_{t} W_{xi} + H_{t-1} W_{hi} + b_{i}) \\
    F_{t} &= \sigma(X_{t}W_{xf} H_{t-1}W_{hf} + b_{f})
\end{align}
$$

from which the cell state can then be computed

$$
    \tilde{C}_{t}= \tanh(X_{t} W_{xc} + H_{t-1}W_{hc} + b_{c})
$$

with the actual update then given by either the candidate cell, if the input gate permits it, or the old cell, if the not-forget gate is on

$$
C_{t} = F_{t} \odot C_{t-1} + I_{t} \odot \tilde{C}_{t}
$$

the hidden state is then computed as a transformed version of the memory cell if the output gate is on

$$
H_{t} = O_{t} \odot \tanh(C_{t})
$$

Visually this then looks like the following:

<center>
    <img src="https://i.imgur.com/Jv06vrG.png" width="450">
</center>

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

There exists a great many variations on this initial architecture, but the core LSTMs' architecture as well as performance have prevailed over time so far. A different approach to sequence generation is the use of causal convolutions with 1-dimensional CNNs. While this approach has shown promise in quite a few practical applications, we view it as beyond the scope of this course.

## Flagship Applications

With so many flagship models of AI these days relying on sequence models, I have compiled a list of a very few of them below for you to play around with. While attention was beyond the scope of this lecture, you can have a look at Lilian Weng's blog post on attention below to dive into it.

* [Opt 175bn Parameter Model for Text Generation](https://opt.alpa.ai)
  * This allows for the free playing with the model by prompting it
* [OpenAI's Dall-E 2](https://openai.com/dall-e-2/)
* [GPT-3](https://openai.com/blog/gpt-3-apps/)
* [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)
* [Cohere AI Blog Posts](https://txt.cohere.ai/generative-ai-part-1/)
* [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

## Further Reading

* [RNN blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [LSTM blog post by Chris Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs)
* [Lilian Weng's blog post on Transformers](https://lilianweng.github.io/posts/2018-06-24-attention/)
