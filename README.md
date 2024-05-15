<!-- ABOUT THE PROJECT -->
# Code and Data for '' Causal Effect Identification in LiNGAM Models with Latent Confounders''.

## Dependencies
Recreate environment:

  ```sh
  conda env create -f requirements.yml
  conda activate experiments
  conda install -c anaconda ipykernel
  python -m ipykernel install --user --name=experiments
  ```

<!-- USAGE EXAMPLES -->
## Usage
### Identifiability Certification
- The code to run the algorithms described in the paper can be found in ```Certification/cert_utils.ipynb```.
- To reproduce the figures in the paper, please see the Jupiter Notebook ```Certification/Identifiability_plots.ipynb```.
### Causal Effect Estimation
- The algorithms for estimation of the causal effects implemented in ```Estimation/src/python``` and ```Estimation/src/matlab```. 
- To reproduce the results one can run the jupyter-notebooks ```Estimation/main_laplace.ipynb``` and ```Estimation/main_exponential.ipynb```. **Please note that**:
  - RICA algorithm is implemented in matlab. Therefore installed matlab is required to reproduce **all** the results. Please change the value of variable ```RUN_MATLAB``` to ```False``` in jupyter-notebooks to reproduce the results of all other algorithms except RICA.

<!-- LICENSE -->
## License

See `LICENSE` for more information.

<!-- Authors -->
## Authors

Anonymous
