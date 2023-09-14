# Jupyter notebooks on deep learning for nanophotonics inverse design

Accompanying the tutorial paper *A newcomer's guide to deep learning for inverse design in nano-photonics*
([arxiv:2307.08618](https://arxiv.org/abs/2307.08618)).

So far, a keras/tensorflow implementation of the tutorial notebooks exists. We demonstrate the full workflow from data generation, and data-processing, over network architecture design and hyperparameter tuning, to an implementation of the above discussed different inverse design approaches.
We use two specific problems for the tutorial notebooks:



## Problem 1: Reflectivity of a layer stack with *PyMoosh*

The first problem used to demonstrate typical deep learning workflow is dielecrtic multi-layer stack with the goal to tailor the reflectivity spectrum.
For the physics calculations these tutorials use [*pyMoosh*](https://github.com/AnMoreau/PyMoosh), the python version of *moosh*, an s-matrix based solver for multilayer optics problems.
For global optimization we use the package *nevergrad*.



## Problem 2: Scattering of dielectric nanostructures

A second problem is used to illustrate the case of structure parametrization via images (here of the geometry top-view). 
This is a typical scenario for many top-down fabricated nano-photonic devices like metasurfaces. 
We demonstrate how a Wasserstein GAN with gradient penalty can be trained on learning a regularized latent description for 2D geometry top-view images.
Using this in combination with a forward predictor model is then demonstrated using global and gradient based optimization of scattering spectrum inverse design, simultaneously for two incident polarizations.
The nano-scattering dataset is created using simulations with the *pyGDM* toolkit (https://homepages.laas.fr/pwiecha/pygdm_doc/).

