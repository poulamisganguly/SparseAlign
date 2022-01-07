[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/poulamisganguly/SparseAlign/HEAD)

# SparseAlign
Example code for the paper "SparseAlign: a super-resolution algorithm for automatic marker localization and deformation estimation in cryo-electron tomography". 

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/poulamisganguly/SparseAlign.git


## Dependencies

The required dependencies for this project are specified in the file `environment.yml`. We used `conda` for package management.


## Usage

The easiest way to try out our code is by running the `example_2d.ipynb` notebook on Binder.

If you'd like to run the code on your machine, start by creating a `conda` virtual environment. 
This can be done by running the following command in the repository folder (where `environment.yml`
is located):

    conda env create -f environment.yml
    
Before running any code you must activate the conda environment:

    conda activate sparse-align
    
Install JupyterLab
```
conda install -c conda-forge jupyterlab
```
Start a JupyterLab instance by running

    jupyter lab --no-browser --port=5678 --ip=127.0.0.1

Open the notebook on your web browser by clicking on the generated link.

## License

All source code is made available under an MIT license.

## Issues and Contributions

If you have any trouble running the code, please open an issue here or contact me at poulami[at]cwi[dot]nl. Thank you for your interest!
