# Quantitative-Raman
Jupyter Notebooks for training and applying a non-linear quantitative model for predicting concentration ratios from Raman intensity ratios.

The QuantModel_Synthetic.ipynb notebook generates synthetic data for exploring the effect of different input parameters on the resulting intensity ratio / concentration ratio curves, and the implications for quantitative model training. The QuantModel_Synthetic.ipynb notebook imports experimentally measured spectra for quantitative model training. The OSTRI_functions.py provides the underlying system of standardised python classes and functions (developed by Dr Joseph Razzell Hollis) for handling Raman & FTIR spectra in Python.

Written by Dr Joseph Razzell Hollis in 2025, last updated on 2026-01-22. OSTRI functions derived from version v0.1.2-alpha, see www.github.com/Jobium/OSTRI/ for updates and documentation.

Any updates to this Notebook will be made available online at www.github.com/Jobium/Quantitative-Raman/

Python code requires Python 3.7 (or higher) and the following packages with minimum versions:
 - numpy 1.19.5
 - pandas 0.22.0
 - matplotlib 2.1.2
 - lmfit 1.0.3
 - scipy 1.5.4
 - scikit-learn 0.24.2
