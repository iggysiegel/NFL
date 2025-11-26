# NFL Prediction Model

## Overview

An extension of the [Glickman-Stern model](https://www.glicko.net/research/nfl.pdf) for NFL predictions, adding quarterback effects to estimate team and QB strength week-to-week. The model is automatically updated every Tuesday morning via GitHub Actions, which stores the latest predictions and fitted model in the repository.

## Installation

### Git LFS (required to fetch weekly models)

This repository uses Git LFS to store the weekly `models/model.npz`. Set up and pull LFS files before running scripts that rely on the fitted model:

```
git lfs install
git lfs pull
```

### Minimal installation — predictions & recommendations only

If you only want to generate the current week’s predictions and betting recommendations (using the pre-fitted model):

```
pip install nflreadpy pandas pyarrow
```

### Full installation — fit the model locally

To fit the full state-space model yourself, create the conda environment provided. This project uses conda because PyMC and some of its dependencies install more reliably via conda channels.

```
conda env create -f environment.yml
```

## Usage

### Generating Weekly Recommendations

Generate betting recommendations for the current week using the pre-fitted model:

```
git lfs pull  # Get the latest model
python -m scripts.update
```

This fetches current lines, loads the fitted model, and generates betting recommendations.

### Fitting the Model

Fit the full state-space model from scratch (requires conda environment, ~20 min runtime depending on hardware):

```
python -m scripts.predict
```

This fits the model using PyMC and saves predictions and the fitted model. For typical use, the automated Tuesday model is sufficient.