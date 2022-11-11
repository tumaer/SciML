# Course Materials to *Scientific Machine Learning for Engineers* WS22-23

All lecture and exercise materials related to this course will be uploaded here.

## Exercise Setup

The easiest way to run the exercise notebooks is through [Google Colab](https://colab.research.google.com/).

If you want to run them locally, we provide this setup tested on Ubuntu 20.04 with CUDA 11.6:

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

