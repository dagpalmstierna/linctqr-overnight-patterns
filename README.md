# Predicting Overnight Reversals in European High-Yield Bond ETFs

This project investigates whether overnight price reversals in European high-yield bond ETFs can be predicted using machine learning. The idea is simple: on days where a high-yield bond ETF closes lower than it opened, does the price tend to bounce back by the next morning's open? And can we build a model that identifies which of those down-days are most likely to reverse?

The full write-up is available here: [Overnight Reversals (PDF)](https://linclund.com/wp-content/uploads/2025/05/Overnight_Reversals.pdf)

## What it does

The pipeline downloads daily price data for the iShares Euro High Yield Corporate Bond ETF (IHYG.L) along with a set of macro and credit market indicators, engineers a large feature set from them, and trains a classifier to predict next-day reversals. Predictions are then backtested to see if the signal has any real trading value.

**Target variable:** A binary label indicating whether the ETF's next-day open is higher than the current close, on days the ETF closed red.

**Features include:**
- Credit spreads (ICE BofA High Yield OAS)
- iTraxx Crossover and iTraxx Europe indices
- ECB yield curve spread
- VSTOXX and VIX volatility indices
- STOXX Europe 600 equity returns
- Technical indicators (RSI, MACD, Bollinger Bands) computed on each of the above
- Rolling statistics (standard deviation, lagged values, percentage changes)

All features go through PCA to reduce dimensionality (retaining 95% of variance) before being passed to the models.

## Models explored

We built and compared three different model types to find what worked best for this problem:

- **Random Forest Classifier** -- Ended up being the best fit. Handles non-linear relationships well, gives feature importance rankings, and performed the most consistently across time-series cross-validation folds. The final version uses a two-pass approach: first train on all PCA components, rank feature importances, then re-train on only the top 25 features.

- **Support Vector Machine (SVM)** -- Tested with RBF and linear kernels using grid search over C and gamma. Worked okay but was more sensitive to hyperparameter choices and didn't generalize as well across folds.

- **LSTM (Long Short-Term Memory)** -- Built a sequence-based deep learning model to see if temporal patterns in the feature set could improve predictions. Interesting to experiment with, but the dataset size (~1600 samples after filtering for red days) wasn't really large enough for an LSTM to shine, and it didn't outperform the Random Forest.

## Class imbalance

Reversal days are the minority class. To deal with this without leaking future information, SMOTE oversampling is applied within rolling 6-month windows on the training data only. Test data is never touched.

## Backtesting

The trading strategy is straightforward: when the model predicts a reversal, buy at close and sell at the next open. Position sizing is scaled inversely by VIX (lower volatility = larger position), and no trades are taken when VIX exceeds 30. The backtest is implemented using the `backtrader` library.

## Project structure

```
predicters/
  dataloader.py      Feature engineering and data loading pipeline
  randomforest.py    Random Forest model with time-series CV
  svm.py             SVM model (alternative approach)
  lstm.py            LSTM model (alternative approach)
  backtest.py        Backtrader strategy definition
  main.py            Entry point -- runs prediction + backtest
  scripts.ipynb      Notebook for interactive exploration and plotting
data/                CSV files for credit spreads, iTraxx, VSTOXX, yield curve
images/              Output plots (confusion matrix, backtest, correlation matrix, PCA)
```

## How to run

```bash
pip install -r requirements.txt
cd predicters
python main.py
```

Note: TA-Lib requires a separate system-level installation. On macOS: `brew install ta-lib`, on Ubuntu: `sudo apt-get install libta-lib-dev`.

Some of the data is fetched live from Yahoo Finance (`yfinance`), so you'll need an internet connection.
