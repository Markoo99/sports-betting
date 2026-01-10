# **Testing Informational Efficiency in NBA Moneyline Betting Markets** 

**Category:** Business & Finance Tools / Statistical Analysis Tools  

---

## Problem Statement or Motivation  

Since a very young age, I have been passionate about sports. I would like to use this project as an opportunity to connect my two main interests: sports and economics. Sports betting markets resemble financial markets, where participants make decisions based on risk awareness and perceived value. Bookmakers set odds according to expected outcomes while incorporating a margin (the “overround”) to ensure profitability and adjusting odds in response to betting volume to balance risk exposure.
This project aims to explore whether sports betting markets are efficient—that is, whether bookmaker-implied probabilities accurately reflect true outcome probabilities. In economic terms, this mirrors the Efficient Market Hypothesis, where prices (or odds) fully incorporate available information. By analyzing historical betting data, I will assess whether systematic inefficiencies exist that could yield consistent positive expected returns.
---

## Planned Approach and Technologies

The analysis will be based on publicly available historical sports betting data, such as NBA results and closing odds from open datasets (e.g., Kaggle’s NBA Betting Odds and Results 2015–2023). The dataset contains match results, odds, and team information—sufficient for building predictive models.
The data will be imported and cleaned using pandas, while numerical computations will be performed with NumPy. Matplotlib and seaborn will be used for visualization.
The main analytical goal is to test market efficiency using statistical and machine learning techniques. Specifically, I will:
Collect and clean historical NBA betting data — convert odds to implied probabilities, handle missing data, and standardize formats.
Compute baseline expected values for different betting strategies.
Train predictive models (e.g., logistic regression and random forest classifiers) to estimate the probability of a team winning.
Compare model predictions to bookmaker-implied probabilities to identify potential biases or inefficiencies.
Conduct hypothesis testing to assess whether the differences are statistically significant.
Visualize cumulative returns, probability calibration curves, and other relevant metrics.
---

## Expected Challenges and How They’ll Be Addressed 

One potential challenge is ensuring the dataset’s completeness and reliability. To address this, the analysis will focus on a single league (NBA) and use verified datasets with historical odds and results. Another issue is statistical validity: models might identify apparent inefficiencies due to random variation rather than genuine patterns. This will be mitigated by using out-of-sample testing, cross-validation, and confidence intervals to ensure robustness and reproducibility.
---

## Success Criteria

The project will be considered successful if it produces a working system capable of importing, analyzing, and visualizing sports betting data while applying data science techniques to evaluate market efficiency. Success will be measured by the ability to train and evaluate predictive models, statistically compare model-derived probabilities to bookmaker odds, and interpret whether deviations suggest inefficiency. Achieving this would demonstrate the application of statistical and machine learning methods to an economically relevant question.
---

## Stretch Goals  

If time permits, Monte Carlo simulation will be used to simulate long term outcomes. 
---
