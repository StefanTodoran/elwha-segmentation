# Environment Management

Environment management is crucial to reproducibility of work and for making your work runnable in hosted containers such as through Binder or Collab. This folder should be used for keeping track of your dependencies. We STRONGLY advise that you use some kind of dependency manager and not just pip install stuff locally!

If you would prefer to work with something other than conda environments, be sure to check out [this](https://mybinder.readthedocs.io/en/latest/using/config_files.html#config-files) page on what kind of files can be used to configure a Binder environment. If you choose to go with a conda environment, then the only file that needs to be edited is `conda/environment.yml`. However if you use anything else some extra setup is going to be required. This is because Binder checks the root folder and the `binder` folder for configuration files. The `binder` folder already contains a symlink pointing to `conda/environment.yml`, but if you choose to use say, `conda/requirements.txt` instead you will need a symlink pointing to that file from the `binder` folder instead.

To create a symlink on windows, use the command `mklink`, of the form `mklink {link_path} {relative_path}`. As an example, the following command is how the existing binder symlink was created:
```
mklink binder\environment.yml ..\conda\environment.yml
```

Although we refer to "conda" environments, we recommend using [mamba](https://github.com/mamba-org/mamba) as a drop in replacement for the `conda` package manager. Mamba performs operations in parallel, which we've found to be faster for creating complex environments involving many packages!

**The only file you should need to edit in this folder is `conda/environment.yml`. This file defines the set of conda-packages needed to render the full website.**

```
conda env create --name geosmart --file=environment.yml
conda activate geosmart
conda env update --file environment.yml --prune
```

Lockfiles ensure that everyone working on this project has an identical development environment, whether working on a personal computer or running on our hosted JupyterHub cloud infrastructure.

If you edit `conda/environment.yml` to change package versions or add new ones, be sure to _re-lock_ the environment by running `./lock-environment.sh`:

```
mamba remove --name geosmart --all
mamba env create --name geosmart --file conda-linux-64.lock.yml
mamba activate geosmart
```
