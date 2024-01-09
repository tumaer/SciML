# Software Infrastructure

There are many ways to get a job done using Python, i.e. different IDEs ([Colab](https://colab.research.google.com/), [VS Code](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/), [Spyder](https://www.spyder-ide.org/), etc.), different libraries (numpy, PyTorch, JAX, etc.), different hardware (CPU, GPU, TPU, parallel CPU/GPU/TPU, etc.). This brief overview should help you find the best setup given your hardware and operating system.

## GPU vs CPU

If your hardware does include a CUDA GPU ([How to know if my GPU supports CUDA](https://askubuntu.com/questions/633176/how-to-know-if-my-gpu-supports-cuda)), then you should probably use it. Note, MacBooks do not have CUDA GPUs and Windows machines require a somewhat different setup than what we will use in the exercises (e.g. [Installing Pytorch with CUDA support on Windows 10](https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e)). Our setup during the exercises is based on Ubuntu 22.04.

If you do not have a CUDA GPU, you should consider using Google Colab with GPU runtime, see [How to use Colab](https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/colab.html#:~:text=What%20is%20Colab%3F&text=It%20allows%20you%20to%20use,the%20session%20after%2012%20hours).

### Why GPU?

1. By moving the core computations to the GPU, the training of deep learning models can be accelerated by more than one order of magnitude.
2. With PyTorch it is relatively straightforward to implement that. You will often see lines like `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')` and `inputs, labels = data[0].to(device), data[1].to(device)`.

## Windows vs Unix (Linux, MacOS)

One of the main disadvantage of Windows in the scope of this class is that the ML community has generally agreed to use Unix, or most often Linux (If you are interested in looking into Linux, try Ubuntu 22.04). This means that a lot of the command line code on Stackoverflow is tested on Linux, and you most probably want to profit from this knowledge base. There are a few things that would need special care working on Windows, and it would be your responsibility to take care of them.

## IDEs

When to use Colab?
- For a quick-and-dirty testing of some functionality
- For demonstration purposes
- If you don't have a CUDA GPU

When not to use Colab?
- Session expire after some time, see [Google Colab session timeout](https://stackoverflow.com/questions/54057011/google-colab-session-timeout). This means that any files other than the notebook itself will be deleted forever.
- The hardware you get for free is not the best one. If you have better hardware, don't use Colab.
- If you want to work on a long-term project, you would need to set up the Colab environment every time for scratch. In contrast, when working locally you could use a virtual environment which you set up once in the beginning.

**Local IDEs (opposed to the web IDE Google Colab)**

Just pick any of the above mentioned: VS Code, PyCharm, Spyder, etc. They have more or less the same functionality and the choise is mainly based on a personal preference rather than software constraints.

## Environments

There are two very popular ways to manage Python environments:
- Conda (Miniconda or Anaconda) - If you see a repository whose dependencies are described in a `environment.yml` file, then conda was used to create the environment
```bash
conda env create --file environment.yml # creates a venv from that .yml file
conda activate <ENV_NAME> # this name is typically in the top-most line of the .yml file
```
- Pip - If you see a repository whose dependencies are described in a `requirements.txt` file, then pip was used to create the environment
```bash
python3 -m venv <ENV_NAME>
source activate <ENV_NAME>/bin/activate
pip install -r requirements.txt
```
There are many tutorials on the internet on how to set up such an environment. Check them for yourself.

### Why virtual environment? 

- Your operating system has its own Python and you probably don't wont to destroy your operating system by changing its default libraries.
- If you work on many projects and each of them requires different libraries, or even worse - different versions of the same library, then you should use one virtual environment per project.

## `.py` vs `.ipynb`

Python files (`.py`) offer more flexibility, powerful debugging options, OOP, etc. They should be the preferred choice. Jupyter notebooks (`.ipynb`) are designed to be used as single projects on themselves or for demonstration purposes. In this course, all exercises are in separate notebooks, and also the practical part of the exam will be a notebook.

### Using Notebooks
If you work on a **Linux or MacOS** device, then you should be able to run notebooks in the same IDE you use for coding, e.g. VS Code.

However, on **Windows** this might not work for multiple reasons, and as we said above we do not provide support for running code on Windows. 

```{admonition} Notebooks on Windows (contributed by Armin Illerhaus)
:class: hint

I didn't find a way to work with Jupyter notebooks in Spyder on Windows. There is one hint on the how to pages that the add-on for Spyder doesn't work with standalone Windows versions. I managed to make it work with the jupyter notebook in the browser (see [that](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/)):

1. Open Anaconda Navigator.
2. Go to environment and open `sciml` environment
3. Right klick on the environment and open terminal
4. `ipython kernel install --user --name=sciml` with user not changed
5. Open Jupiter in any browser and select as Kernel the `sciml` environment
```

