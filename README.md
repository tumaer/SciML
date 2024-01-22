# *Scientific Machine Learning for Engineers*

All lecture and exercise materials related to this course will be uploaded here.

## Exercise Setup

The easiest way to run the exercise notebooks is through [Google Colab](https://colab.research.google.com/).

If you want to run them locally, we provide this setup tested on Ubuntu 22.04 and macOS 14:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Admin

### Jupyter Book

To build the book:

```bash
jb build .
```

To remove all files in the `_build` dir:

```bash
jb clean .
```

### GitHub Pages

First, you need to `pip install ghp-import`. Then from the `master` branch push to GitHub Pages via:

```
ghp-import -n -p -f _build/html
```

More info: https://jupyterbook.org/en/stable/publish/gh-pages.html

### MyST Markdown

These are some instructions on how to edit the Jupyter book. 

**Equations**

Name equations uniquely to enable referencing them across the whole book, e.g.

```

$$y = ax + b$$ (weltgleichung)

Die Weltgleichung {eq}`weltgleichung`.
```

> Do not forget to include an emptry line before and after an equation, unless it is within the text.

**Figures**

For a proper/referencable figure, use

``````
```{figure} ../imgs/my_figure.png
---
width: 500px
align: center
name: my_figure_name
---
My figure (Source: {cite}`bishop2006`).
```
``````

And then refer to it using 

```
this style: {numref}`my_figure_name`.
```

**Referencing**

Cite Bolstad by using 

```
this stype: {cite}`bolstad2009`.
```

**Indentation**

If e.g. an equation should be within a bullet point, it should be indented respectively. Otherwise, it breaks the bullet list.

```
Numbered list:

1. First point: some stuff following equation
    
    $$y = ax + b$$ (weltgleichung2)

2. Some more stuff

```

**Misc**

- To put highlighted content, use `> some stuff` on a new line.
- To include a simple markdown URL use `[this](https://x.com)`.
- To refer to other files use `[Some description](./lecture/x.md)`, or to use their own heading, use `[](./lecture/x.md)`.
- Inline equations as `$y=f(x)$`.
- All functionalities from `.md` apply also to `.ipynb`.