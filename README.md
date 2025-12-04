# NFL Prediction Model

## Overview

An extension of the [Glickman-Stern model](https://www.glicko.net/research/nfl.pdf) for NFL predictions, adding quarterback effects to estimate team and QB strength week-to-week.

## Installation

To fit the full state-space model yourself, create the conda environment provided. This project uses conda because PyMC and some of its dependencies install more reliably via conda channels.

```bash
conda env create -f environment.yml
```

## Usage

Fit the full state-space model from scratch (~20 min runtime depending on hardware):

```bash
python -m scripts.predict
```

Generate betting recommendations for the current week using the fit model:

```bash
python -m scripts.update
```
