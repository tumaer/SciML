# SciML Software Infrastructure

There are many ways to get a job done using Python, i.e. different IDEs ([Colab](https://colab.research.google.com/), [VS Code](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/), [Spyder](https://www.spyder-ide.org/), etc.), different libraries (numpy, PyTorch, JAX, etc.), different hardware (CPU, GPU, TPU, parallel CPU/GPU/TPU, etc.). This brief overview should help you find the best setup for given your hardware and operating system.

## GPU vs CPU

If your hardware does include a CUDA GPU ([How to know if my GPU supports CUDA](https://askubuntu.com/questions/633176/how-to-know-if-my-gpu-supports-cuda)), then you should probably use it. Note, MacBooks do not have CUDA GPUs and Windows machines require a somewhat different setup than we will present here (e.g. [Installing Pytorch with CUDA support on Windows 10](https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e); our setup during the exercises is based on Ubuntu 20.04., e.g. [this](https://github.com/arturtoshev/test_torch_cuda) tutorial.

If you do not have a CUDA GPU, you should consider using Google Colab with GPU runtime, see [How to use Colab](https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/colab.html#:~:text=What%20is%20Colab%3F&text=It%20allows%20you%20to%20use,the%20session%20after%2012%20hours).

### Why GPU?

1. By moving the core computations to the GPU, the training of deep learning models can be accelerated with up to one order of magnitude.
2. With PyTorch it is relatively straightforward to implement that. You will often see lines line `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')` and `inputs, labels = data[0].to(device), data[1].to(device)`.

## Windows vs Unix (Linux, MacOS)

One of the main disadvantage of Windows in the scope of this class is that the ML community has generally agreed to use Unix, or most often Linux (If you are interested in looking into Linux, try Ubuntu 22.04). This means that a lot of the command line code on stackoverflow is tested on Linux, and you most probably want to profit from this knowledge base. There are a few things that would need special care working on Windows, and it would be your responsibility to take care of them.

## IDEs

When to use Colab?
- For a quick-and-dirty testing of some functionality
- For demonstration purposes
- If you don't have a CUDA GPU

Why not to use Colab?
- Session expire after some time, see [Google Colab session timeout](https://stackoverflow.com/questions/54057011/google-colab-session-timeout). This means that any files other than the notebook itself will be deleted forever.
- The hardware you get for free is not the best one. If you have better hardware, don't use Colab.
- If you want to work on a long-term project, you would need to set up the environment every time for scratch. Locally you could use a virtual environment

**Local IDEs (opposed to the web IDE Google Colab)**

Just pick any of the above mentioned: VS Code, PyCharm, Spyder, etc. They have more or less the same functionality and the choise is mainly based on a personal preference rather that software constraints.

## Environments

There are two very popular ways to manage Python environments:
- Conda (Miniconda or Anaconda) - If you see a repository whose dependencies are described in a `environment.yml` file, then conda was used to create the environment
```bash=
conda env create --file environment.yml
conda activate <ENV_NAME> # this name is typically in the top-most line of the .yml file
```
- Pip - If you see a repository whose dependencies are described in a `requirements.txt` file, then pip was used to create the environment
```bash=
python3 -m venv <ENV_NAME>
source activate <ENV_NAME>/bin/activate
pip install -r requirements.txt
```
There are many tutorial on the internet how to set up such an environment. Check them for yourself.

### Why virtual environment? 

- Your operating system has its own Python and you probably don't wont to destroy your operating system by changing its default libraries
- If you work on many projects and each of them requires different libraries, or even worse - different versions of the same library, then you should use one virtual environment per project.

## `.py` vs `.ipynb`

Python files (`.py`) offer more flexibility, powerful debugging options, OOP, etc. They should be the preferred choice. Notebooks are designed to be used as single projects on themselves or for demonstration purposes.