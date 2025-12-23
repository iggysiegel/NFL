# NFL Prediction Model

## Overview

An extension of the [Glickman-Stern model](https://www.glicko.net/research/nfl.pdf) for NFL predictions, adding quarterback effects.

- [Visit the website](https://iggysiegel.github.io/NFL/)

## Installation

To view this week's predictions, check out the website linked above.

If you want to fit the full state-space model yourself, create the conda environment provided. This project uses conda because PyMC and some of its dependencies install more reliably via conda channels.

```bash
conda env create -f environment.yml
```

## Usage

Fit the full state-space model from scratch (~15 min runtime depending on hardware):

```bash
python -m scripts.predict
```

Generate recommendations for the current week using the fitted model:

```bash
python -m scripts.update
```
