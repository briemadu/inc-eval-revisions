#!/bin/bash

conda create --name inc-eval-revisions python=3.9
conda activate inc-eval-revisions
conda install pandas=2.0.3
conda install seaborn
conda install -c conda-forge scikit-learn
conda install Jinja2

mkdir figures
mkdir preprocessed