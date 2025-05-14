# Copulas and Dynamic Hedging Techniques in Pairs Trading

This project implements a pairs trading strategy using statistical dependence structures (copulas) and dynamic hedge ratio estimation. It was developed as part of the IAQF 2023 challenge submission.

## 🧠 Abstract

Pairs trading is a relative value strategy that identifies two funds or indices with similar characteristics that have diverged from their historical price relationship. The strategy involves buying the undervalued asset and shorting the overvalued one, betting on mean reversion.

In this project, we:
- Use **Kendall's Tau** and **sum of squared differences** to select the most correlated asset pairs
- Estimate **joint distributions** using **copula functions**
- Generate **entry and exit signals** based on quantile thresholds
- Implement a **dynamic hedge ratio** update mechanism
- Conduct **backtesting and parameter updates**

## 📁 Files

- `IAQF_2023_solution.pdf`: Project summary and methodology submitted for the IAQF competition
- `MVPS_IAQF.ipynb`: Main Jupyter notebook for pair selection, copula fitting, and backtesting
- `copulas.R`: R script for copula density estimation and visualization
- `IAQF.py`: Auxiliary Python script for signal generation and trade simulation

## 📊 Methods

- **Statistical Measures**: Kendall Tau, SSD (Sum of Squared Differences)
- **Copula Families**: Gaussian, Student-t, Clayton, Gumbel
- **Quantile-Based Signal Triggers**
- **Rolling Hedge Ratio Estimation**

## 🧪 Results

The strategy shows how copula-based modeling can more flexibly capture non-linear dependencies between asset pairs, improving signal generation beyond simple linear correlation.

---
