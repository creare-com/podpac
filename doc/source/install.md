# Installation Instructions

At the moment, a full Windows 10 installation of PODPAC can be downloaded from our [PODPAC-S3-Bucket](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip). 

For custom installations, commandline installation procedures are described below. 

These instructions assume you are using the [Anaconda Python Distribution](https://www.anaconda.com/), and have [git](https://git-scm.com/) installed.

## Window 10 Installation
A full Windows 10 Installation of PODPAC can be downloaded from [here](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip). To use it, extract the zip file in a folder on your machine. Then:

* To open the example notebooks, run the `run_podpac_jupyterlab.bat` script, by double-clicking the icon
    * This will open up a Windows command prompt, and launch a JupyterLab notebook in your default web browser
        * Older browsers may not support JupyterLab, as such the url with the token can be copied and pasted from the Windows command prompt that was launched
    * To close the notebook, close the browser tab, and close the Windows console
* To run an IPython session:
    * Open a Windows command prompt in the unzipped folder
    * Set up the appropriate Windows environment by running the `set_local_conda_path.bat` script
    * Set up absolute paths used within Anaconda to point to your local installation path by running the `fix_hardcoded_absolute_paths.bat`
    
    * Activate the PODPAC Python environment by running the `activate_podpac_conda_env.bat` script
    * Open an IPython console by typing `ipython` and hitting enter

## Commandline Installation
### Conda Environment

We recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda   # installs all `anaconda` packages
$ conda activate podpac  # Windows
$ source activate podpac # Linux / Mac
```

### Installation Instructions for Users
***Note***: We plan to improve this installation process after reaching version 1.0

#### Clone the Repository

Clone the [podpac repository](https://github.com/creare-com/podpac) onto your machine:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
```

Checkout the latest or desired version:

```bash
$ git checkout tags/<version> release/<version>  
```
For example to checkout version 0.2.0 use:
```bash
$ git checkout tags/0.2.0 release/0.2.0
```

#### Installing dependencies
PODPAC's dependencies are automatically installed through `pip` when `podpac` is installed. Some dependencies are more difficult to install on certain systems. 

In particular, some users may experience issues installing [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html#installing-with-anaconda) (included in the `datatype`, `all`, and `devall` installations). If you encounter issues, we recommend trying to install from the **conda-forge** channel. Depending on your platform, this may be simpler than letting `podpac` install `rasterio` using `pip`:

```bash
$ conda install rasterio --channel conda-forge
```

#### Installing podpac

After cloning the repository to your computer, install `podpac` using `pip`. PODPAC comes with a number of optional dependency packages which can be installed alongside PODPAC. These packages include:

* datatype
* aws
* algorithms
* notebooks
* esri
* dev
* all
    * includes `datatype`, `aws`, `algorithms`, `notebook`, and `esri`
* devall
    * includes `all` and `dev`

For example, to install podpac with only the core dependencies, execute the following from the `podpac` folder

```bash
$ pip install .
```

To install podpac with all the optional dependencies, use:

```bash
$ pip install .[all]
```

If you experience problem with dependencies, see the [Creating the Windows Installation PODPAC Conda environment](#creating-the-windows-installation-podpac-conda-environment) section.

If you are installing the optional notebook dependencies, you also need to run the following commands to install the `JupyterLab` plugins we use:
```bash
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter labextension install jupyter-leaflet
$ jupyter labextension install jupyter-matplotlib
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter lab build
$ python -m ipykernel install --user
```

#### Running example notebooks
To run the PODPAC example notebooks, start JupyterLab in the `doc/notebooks` directory of PODPAC

```bash
$ cd doc/notebooks
$ jupyter lab
```
Open a notebook and select `Run` from the top menu, followed by `Run All`.
You may be prompted to enter user EarthData login credentials to access NASA data. 

### Installation Instructions for Developers

#### Clone the Repository

Clone the [podpac repository](https://github.com/creare-com/podpac) onto your machine:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
```

The `master` branch is intented to be somewhat stable with working code. For bleeding edge, checkout the `develop` branch instead:

```bash
$ git checkout -b develop origin/develop
```

#### Installing podpac

After cloning the repository to your computer, install `podpac` in development mode using pip:

```bash
# Install podpac with only the core dependencies
$ pip install -e .

# development dependencies and all other dependencies
$ pip install -e .[devall]
```

Some users may experience issues installing [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html#installing-with-anaconda) (included in the `datatype`, `all`, and `devall` installations). If you encounter issues, we recommend trying to install from the **conda-forge** channel. Depending on your platform, this may be simpler than letting `podpac` install `rasterio` using pip:

```bash
$ conda install rasterio --channel conda-forge
```

## Creating the Windows Installation PODPAC Conda environment
This section describes the process used to create the [PODPAC Window 10 Installation](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip).

These instructions only assume that you already have [git](https://git-scm.com/) installed on your Windows 10 machine. 

* Install miniconda to a `<root_folder>`\miniconda on a Windows machine
    * Install for "Just the current user"
    * Do not register Python to the operating system
* Open a Windows command prompt in the `<root_folder>` 
* Clone podpac and set up the conda environment
```bash
> git clone https://github.com/creare-com/podpac.git
> cd podpac
> git checkout tags/<version> release/<version>  # as of writing, the <version> is 0.2.0
> cd ..
> copy podpac\dist\local_Windows_install\* .
> set_local_conda_path.bat
# Verify path is set correctly
> where conda
<root_folder>\miniconda\Library\bin\conda.bat  # Should be the first entry
<root_folder>\miniconda\Scripts\conda.exe      # Should be the second entry
...                                            # May have additional entries
```
* Set up the podpac environment and install dependencies
```bash
conda create -n podpac python=3
# Install core dependencies
conda install matplotlib>=2.1 numpy>=1.14 pint>=0.8 scipy>=1.0 traitlets>=4.3 xarray>=0.10 ipython
# Install dependencies for handling various file datatype
conda install rasterio>=0.36 -c conda-forge
conda install beautifulsoup4>=4.6 h5py>=2.7 lxml>=4.2 pydap>=3.2 requests>=2.18 
# Install dependencies for AWS
conda install awscli>=1.11 boto3>=1.4
# Install dependencies for algorithm nodes
conda install numexpr>=2.6
# Install dependencies for JupyterLab and Jupyter Notebooks
conda install jupyterlab ipyleaflet ipywidgets ipympl nodejs -c conda-forge
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-leaflet
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
jupyter lab build
python -m ipykernel install --user
conda clean -a -y
```
* To run a `JupyterLab` sessions in the `<root_folder>\podpac\doc\notebooks` directory, double-click on the `run_podpac_jupyterlab.bat`. This will launch a browser window in the folder where PODPAC keeps its example notebooks.
* To run an IPython console: Open up a Windows command prompt in `<root_folder>`
```bash
set_local_conda_path.bat
fix_hardcoded_absolute_paths.bat
activate_podpac_conda_env.bat
ipython
```