# Promotional Demand Forecasting — Akuret Solutions

## Technical Approach Document | Prepared by Saim Malik

---

> **Company:** Akuret Solutions — AI-Powered Shelf Intelligence for Grocery Retail
> **Problem:** Promotional demand forecasting using transactional data
> **Constraint:** Promotions announced 1-2 weeks in advance; predict demand lift accurately
> **Goal:** Automate the forecasting pipeline with an AI agent

---

## Table of Contents

| # | Section |
|---|---------|
| 1 | [The Business Problem](#1-the-business-problem) |
| 2 | [Types of Time Series Problems](#2-types-of-time-series-problems) |
| 3 | [The Data — What We're Working With](#3-the-data--what-were-working-with) |
| 4 | [Feature Engineering — The Real Work](#4-feature-engineering--the-real-work) |
| 5 | [Model Selection Framework](#5-model-selection-framework) |
| 6 | [Tier 1: Classical Statistical Models](#6-tier-1-classical-statistical-models) |
| 7 | [Tier 2: Machine Learning Models](#7-tier-2-machine-learning-models) |
| 8 | [Tier 3: Deep Learning Models](#8-tier-3-deep-learning-models) |
| 9 | [Model Comparison Matrix](#9-model-comparison-matrix) |
| 10 | [Clustering as Preprocessing](#10-clustering-as-preprocessing) |
| 11 | [Evaluation Metrics for Forecasting](#11-evaluation-metrics-for-forecasting) |
| 12 | [The Agent Architecture](#12-the-agent-architecture) |
| 13 | [Complete Pipeline — End to End](#13-complete-pipeline--end-to-end) |
| 14 | [Time Series Databases & Tools](#14-time-series-databases--tools) |
| 15 | [Key Concepts You Must Know](#15-key-concepts-you-must-know) |
| 16 | [Common Pitfalls in Demand Forecasting](#16-common-pitfalls-in-demand-forecasting) |
| 17 | [Recommended Approach for Akuret](#17-recommended-approach-for-akuret) |
| 18 | [Interview Q&A — Forecasting](#18-interview-qa--forecasting) |

---

## 1. The Business Problem

### What Akuret needs

A grocery retailer announces a promotion — "Buy 1 Get 1 Free on Coca-Cola" starting in 2 weeks. The retailer needs to answer one critical question:

**"How many extra units should we stock?"**

- **Order too few** → out-of-stock during promo → lost sales, angry customers, wasted ad spend
- **Order too many** → excess inventory → waste (perishables expire), tied-up capital, markdowns

This is a **demand forecasting** problem, specifically **promotional demand forecasting** — predicting the sales uplift (or "lift") caused by a promotion above the baseline demand.

### Why this is NOT classification

| Problem Type | What it predicts | Example |
|---|---|---|
| **Classification** | A category/label | "Is this customer likely to churn? Yes/No" |
| **Regression** | A continuous number | "How many units will sell next week?" |
| **Forecasting** | A number over time | "What will daily sales look like for the next 14 days?" |
| **Clustering** | Groups of similar items | "Which stores behave similarly?" (preprocessing step) |

**Akuret's problem = Forecasting (a type of regression over time)**

The core output is a **number** — predicted units to sell per SKU per store per day during the promotional period. Not a label, not a category.

### The math behind promotional lift

```
Promotional Demand = Baseline Demand × (1 + Lift Factor)

Where:
  Baseline Demand = what would sell WITHOUT a promotion
  Lift Factor     = the multiplier caused by the promotion (e.g., 2.5x for BOGO)

Example:
  Baseline: 100 units/week of Coke
  BOGO promotion → historical lift factor = 2.5x
  Predicted promotional demand = 100 × 2.5 = 250 units/week
  Extra units to order = 250 - 100 = 150 units
```

But it's rarely this simple — lift factors vary by store, product, season, day of week, competitor actions, and promotion type.

---

## 2. Types of Time Series Problems

### The spectrum of time series tasks

| Type | Description | Example | Models |
|---|---|---|---|
| **Univariate forecasting** | Predict future values from a single variable's history | "Predict next week's Coke sales from past Coke sales" | ARIMA, ETS, Prophet |
| **Multivariate forecasting** | Predict using multiple variables | "Predict Coke sales using price, weather, promos, competitor data" | XGBoost, TFT, LSTM |
| **Probabilistic forecasting** | Predict a distribution, not a point | "Sales will be 200-280 units (90% confidence)" | DeepAR, Prophet, Bayesian models |
| **Hierarchical forecasting** | Forecast at multiple levels that must reconcile | "Store → Region → National forecasts must add up" | Hierarchical reconciliation, MinT |
| **Intermittent demand** | Products that sell sporadically | "Specialty item: 0, 0, 3, 0, 0, 1, 0, 0, 5" | Croston's method, INARMA |
| **Causal forecasting** | Understand what CAUSED the change, not just predict | "How much did the promo contribute vs the holiday?" | CausalImpact, Bayesian structural |

**Akuret's problem is primarily multivariate + probabilistic forecasting** — multiple input features (promo type, price, history, store attributes) producing a demand prediction with confidence intervals.

### Stationarity — the most important concept

A time series is **stationary** if its statistical properties (mean, variance) don't change over time. Most forecasting models assume stationarity or require you to transform the data to achieve it.

**Non-stationary patterns to watch for:**
- **Trend** — sales steadily increasing or decreasing over time
- **Seasonality** — repeating patterns (weekly, monthly, yearly)
- **Structural breaks** — sudden permanent shifts (new competitor opens, COVID)

**How to make data stationary:**
- **Differencing** — subtract previous value: `y'(t) = y(t) - y(t-1)` removes trend
- **Seasonal differencing** — subtract value from same season last year
- **Log transformation** — stabilizes variance when it grows with the level
- **ADF test** (Augmented Dickey-Fuller) — statistical test for stationarity

### Decomposition — breaking down a time series

```
Time Series = Trend + Seasonality + Residual     (additive)
Time Series = Trend × Seasonality × Residual     (multiplicative)

Where:
  Trend       = long-term direction (growth/decline)
  Seasonality = repeating cycles (weekly, monthly, yearly)
  Residual    = what's left (noise + unexplained variation + PROMO EFFECTS)
```

The promotion creates a **spike** in the residual. Our job is to predict the size of that spike.

---

## 3. The Data — What We're Working With

### Transactional data (the spreadsheet Kevin mentioned)

| Field | Description | Example |
|---|---|---|
| `date` | Transaction date | 2024-03-15 |
| `store_id` | Which store | STORE_042 |
| `sku` | Product identifier | COKE_12OZ_REG |
| `units_sold` | Quantity sold | 48 |
| `revenue` | Dollar amount | $57.60 |
| `unit_price` | Selling price | $1.20 |
| `regular_price` | Non-promo price | $1.50 |
| `promo_flag` | Was there a promotion? | 1 (yes) |
| `promo_type` | Type of promotion | BOGO, PCT_OFF, BUNDLE |
| `promo_depth` | Discount percentage | 50% (for BOGO) |
| `category` | Product category | Beverages > Carbonated > Cola |
| `brand` | Brand name | Coca-Cola |

### Additional data sources

| Source | What it adds | Why it matters |
|---|---|---|
| **Calendar** | Holidays, day of week, pay days | Black Friday spike, weekend patterns |
| **Weather** | Temperature, precipitation | Ice cream sales spike in heat |
| **Competitor** | Competitor promotions | Pepsi promo cannibalizes Coke baseline |
| **Store attributes** | Location, size, demographics | Urban vs rural response to promos |
| **Planogram** | Shelf position, facing count | End-cap placement amplifies promo lift |
| **Inventory** | Current stock levels, lead times | Constrain forecast to what's orderable |

### Data quality issues to expect

- **Missing data** — stores didn't report some days, products temporarily delisted
- **Censored demand** — if a product was out of stock, recorded sales = 0 but actual demand was higher (stockout bias)
- **Cannibalization** — Coke promo increases Coke sales but decreases Pepsi sales
- **Halo effect** — Coke promo increases chip sales too (cross-category lift)
- **Forward buying** — customers stockpile during promo, so post-promo sales drop

---

## 4. Feature Engineering — The Real Work

### Time-based features

| Feature | How to compute | Why it matters |
|---|---|---|
| `day_of_week` | Mon=0, Sun=6 | Weekend patterns |
| `week_of_year` | 1-52 | Seasonal cycles |
| `month` | 1-12 | Monthly seasonality |
| `is_holiday` | Binary flag | Holiday demand spikes |
| `days_to_holiday` | Count | Pre-holiday shopping ramp |
| `is_payday` | Binary flag | Spending cycles |
| `is_weekend` | Binary flag | Weekend vs weekday |

### Lag features (historical patterns)

| Feature | What it captures |
|---|---|
| `sales_lag_7` | Same day last week |
| `sales_lag_14` | Same day 2 weeks ago |
| `sales_lag_365` | Same day last year |
| `rolling_mean_7` | Average of last 7 days |
| `rolling_mean_28` | Average of last 28 days (monthly trend) |
| `rolling_std_7` | Volatility of last 7 days |
| `sales_trend` | Slope of recent sales (growing/declining) |

### Promotion features (the critical ones)

| Feature | What it captures |
|---|---|
| `is_promo` | Binary — is there a promotion active? |
| `promo_type_encoded` | One-hot: BOGO, PCT_OFF, BUNDLE, LOYALTY |
| `promo_depth` | Discount percentage (0-100) |
| `promo_duration_days` | How long the promo lasts |
| `days_into_promo` | Day 1 of promo vs day 7 (fatigue effect) |
| `days_since_last_promo` | Recency of last promotion on this product |
| `promo_frequency` | How often this product is promoted |
| `historical_lift` | Average lift from past promos of same type |
| `competitor_promo` | Is competitor running a similar promo? |
| `promo_position` | End-cap, in-aisle, front-of-store |

### Cross-product features

| Feature | What it captures |
|---|---|
| `category_promo_count` | How many other products in category are on promo |
| `brand_promo_count` | How many other products from same brand on promo |
| `cannibalization_index` | Historical % of sales stolen from substitutes |
| `halo_index` | Historical % of uplift on complementary products |

---

## 5. Model Selection Framework

### How to choose the right model

The selection depends on three factors:

**Factor 1: Data volume**
- < 100 data points per series → Classical statistical models
- 100-10,000 per series → ML models (XGBoost, LightGBM)
- 10,000+ across many series → Deep learning models

**Factor 2: Number of features**
- 1-2 features → Univariate (ARIMA, ETS, Prophet)
- 5-50 features → ML models (handle feature interactions)
- 50+ features including embeddings → Deep learning

**Factor 3: Number of time series**
- Single product/store → Classical (one model per series)
- Hundreds of product/store combinations → ML or deep learning (learn across series)
- Thousands+ → Deep learning (global model, transfer learning across series)

### The model selection decision tree

```
START
  │
  ├─ Do you have < 2 years of weekly data per SKU?
  │    YES → Prophet or ETS (robust with limited data)
  │    NO ↓
  │
  ├─ Do you have rich features (promo type, price, weather)?
  │    NO → SARIMA or Prophet (univariate is fine)
  │    YES ↓
  │
  ├─ Do you need to forecast 1000+ SKU/store combinations?
  │    NO → XGBoost/LightGBM (fast, interpretable)
  │    YES ↓
  │
  ├─ Do you have 50K+ training samples across all series?
  │    NO → XGBoost/LightGBM with cross-series features
  │    YES ↓
  │
  └─ Use TFT or DeepAR (global model, handles scale)
```

---

## 6. Tier 1: Classical Statistical Models

### ARIMA / SARIMA

**What it is:** AutoRegressive Integrated Moving Average. The workhorse of time series forecasting.

**Components:**
- **AR (AutoRegressive):** Current value depends on past values. `y(t) = c + φ₁y(t-1) + φ₂y(t-2) + ...`
- **I (Integrated):** Differencing to make series stationary. `d=1` means first difference.
- **MA (Moving Average):** Current value depends on past forecast errors. `y(t) = c + θ₁ε(t-1) + θ₂ε(t-2) + ...`
- **S (Seasonal):** Adds seasonal AR, I, MA terms. `SARIMA(p,d,q)(P,D,Q,s)`

**Parameters:** `ARIMA(p, d, q)` where p=AR order, d=differencing order, q=MA order.

**Strengths:**
- Well-understood, extensively studied
- Fast to train and predict
- Good for clean, single-series data
- Interpretable parameters
- Works with small datasets

**Weaknesses:**
- Univariate only (can't use promo features directly)
- Assumes linear relationships
- Struggles with multiple seasonalities
- Requires manual parameter tuning (or auto_arima)
- One model per SKU/store → doesn't scale to thousands

**When to use for Akuret:** Baseline model. Run ARIMA on historical sales to establish the "no-promotion" baseline, then measure promotional lift against it.

**Python:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(sales_data, order=(1,1,1), seasonal_order=(1,1,1,52))
results = model.fit()
forecast = results.forecast(steps=14)
```

### Exponential Smoothing (ETS / Holt-Winters)

**What it is:** Forecasting by giving exponentially decreasing weights to older observations. Recent data matters more.

**Three variants:**
- **Simple ES:** Level only (flat forecast)
- **Holt (Double ES):** Level + Trend
- **Holt-Winters (Triple ES):** Level + Trend + Seasonality

**Strengths:**
- Very fast, minimal computation
- Handles trend and seasonality well
- Works with small datasets
- Easy to implement and explain
- Good for thousands of SKUs (fast enough to run individually)

**Weaknesses:**
- Univariate only
- Can't incorporate external features (promos, weather)
- Assumes regular seasonal patterns
- No uncertainty estimation (point forecasts only)

**When to use for Akuret:** Quick baseline across all SKUs. If you have 10,000 SKU/store combinations, ETS can run on all of them in seconds.

**Python:**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(sales, trend='add', seasonal='add', seasonal_periods=52)
results = model.fit()
forecast = results.forecast(14)
```

### Prophet (Meta/Facebook)

**What it is:** Decomposable time series model designed for business forecasting. Handles missing data, outliers, and holidays natively.

**Components:**
```
y(t) = trend(t) + seasonality(t) + holidays(t) + regressors(t) + error(t)
```

**Key feature for Akuret:** The `regressors` term — you can add promotion as an external regressor!

```python
model = Prophet()
model.add_regressor('is_promo')
model.add_regressor('promo_depth')
model.add_regressor('is_holiday')
model.fit(df)
forecast = model.predict(future_df)  # future_df includes known promo schedule
```

**Strengths:**
- Handles multiple seasonalities (daily, weekly, yearly)
- Robust to missing data and outliers
- Can add promotions as regressors
- Automatic changepoint detection (trend shifts)
- Built-in uncertainty intervals
- Easy to use, minimal tuning needed

**Weaknesses:**
- Doesn't capture complex feature interactions
- Each regressor has a linear effect (can't model "BOGO on Coke during summer" interaction)
- One model per series (doesn't learn across products)
- Can overfit on short time series
- Slower than ETS for large-scale deployment

**When to use for Akuret:** Strong first production model. Handles the "known future promo" scenario natively and gives confidence intervals for order decisions.

---

## 7. Tier 2: Machine Learning Models

### XGBoost / LightGBM (Gradient Boosted Trees)

**What it is:** Ensemble of decision trees trained sequentially, each correcting the previous one's errors. The most common production model for tabular data forecasting.

**How it works for time series:**
1. Transform time series into a supervised learning problem using lag features
2. Each row = one data point: `[lag_1, lag_7, lag_365, day_of_week, is_promo, promo_type, ...] → units_sold`
3. Train the model on historical rows, predict future rows where promo features are known

**Strengths:**
- Handles dozens of features naturally
- Captures non-linear relationships and feature interactions
- "BOGO on Coke in summer in urban stores" — trees split on all these naturally
- Fast training and inference
- Feature importance tells you what matters
- Works well with moderate data sizes
- Extremely well-supported (scikit-learn, xgboost, lightgbm)

**Weaknesses:**
- Requires manual feature engineering (lag features, rolling stats)
- Can't extrapolate beyond training range (trees can only predict values they've seen)
- Doesn't inherently understand time ordering — you must encode it via features
- Risk of data leakage if lag features aren't carefully constructed
- One global model or per-group models — needs thoughtful grouping strategy

**When to use for Akuret:** **Primary production model.** This is what most retail forecasting systems actually use. The feature engineering is the hard part, but the model itself is proven.

**Python:**
```python
import lightgbm as lgb

features = ['sales_lag_7', 'sales_lag_14', 'rolling_mean_7',
            'day_of_week', 'month', 'is_promo', 'promo_type',
            'promo_depth', 'historical_lift', 'store_cluster']

model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X_train[features], y_train)
predictions = model.predict(X_test[features])
```

### Random Forest

**What it is:** Ensemble of decision trees trained independently on random subsets of data. Each tree votes, majority wins.

**Strengths:**
- More robust to overfitting than XGBoost
- No sequential dependency (parallelizable)
- Good baseline ML model
- Feature importance built-in

**Weaknesses:**
- Generally lower accuracy than XGBoost/LightGBM for tabular data
- Same feature engineering requirements
- Slower inference than boosted trees

**When to use for Akuret:** Quick sanity-check model. If Random Forest performs close to XGBoost, your features are doing the heavy lifting (good). If XGBoost is much better, the boosting is capturing complex interactions you should investigate.

---

## 8. Tier 3: Deep Learning Models

### LSTM / GRU (Recurrent Neural Networks)

**What it is:** Neural networks with memory gates that process sequential data. LSTM (Long Short-Term Memory) remembers long-range patterns; GRU (Gated Recurrent Unit) is a simplified, faster variant.

**How it works for demand forecasting:**
```
Input sequence: [week_1_features, week_2_features, ..., week_N_features]
                     ↓
              LSTM processes sequence
                     ↓
              Output: predicted_sales_week_N+1
```

**Strengths:**
- Naturally handles sequential data
- Can learn long-range temporal dependencies
- Handles multivariate input
- No manual lag feature engineering needed

**Weaknesses:**
- Needs large datasets (10K+ samples)
- Slow to train
- Hard to interpret ("why did it predict 250 units?")
- Prone to overfitting without careful regularization
- Vanishing gradient issues in very long sequences

**When to use for Akuret:** When you have substantial historical data and the classical/ML models plateau. Not the first model to try.

### N-BEATS (Neural Basis Expansion Analysis for Time Series)

**What it is:** A pure deep learning architecture specifically designed for time series. Uses a stack of fully connected networks with backward and forward residual connections.

**Key innovation:** Interpretable variant decomposes forecast into trend and seasonality components.

**Strengths:**
- State-of-the-art on many benchmarks
- No feature engineering needed
- Interpretable variant available
- Fast inference

**Weaknesses:**
- Univariate only (standard version)
- Needs large datasets
- Less proven in production retail settings

**When to use for Akuret:** When you want a deep learning baseline that doesn't require external features. Good for the "pure pattern" component of the forecast.

### DeepAR (Amazon)

**What it is:** Autoregressive probabilistic forecasting model. Predicts a probability distribution, not just a point estimate.

**Key innovation:** Trains one global model across ALL time series. Learns patterns from high-volume products and transfers knowledge to low-volume ones.

**Output:**
```
Instead of: "Coke will sell 250 units"
DeepAR says: "Coke will sell 200-280 units (90% confidence)"
```

**Strengths:**
- Probabilistic — gives confidence intervals for inventory decisions
- Global model — learns across all products/stores simultaneously
- Handles cold-start (new products with limited history)
- Available in Amazon SageMaker (your experience!)
- Handles covariates (promotion features)

**Weaknesses:**
- Requires AWS infrastructure (SageMaker) for easiest deployment
- Needs substantial data across many series
- Black box — hard to explain individual predictions
- Training is compute-intensive

**When to use for Akuret:** Strong production candidate given your SageMaker experience. Particularly valuable because retailers need confidence intervals ("order between X and Y units"), not just point estimates.

**SageMaker deployment:**
```python
from sagemaker import estimator

deepar = sagemaker.estimator.Estimator(
    image_uri=sagemaker.image_uris.retrieve("forecasting-deepar", region),
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/output'
)
deepar.set_hyperparameters(
    time_freq='W',
    prediction_length=2,  # 2 weeks ahead
    context_length=52,    # 1 year of context
    num_cells=40,
    num_layers=2
)
deepar.fit({'train': train_data_s3, 'test': test_data_s3})
```

### Temporal Fusion Transformer (TFT)

**What it is:** Google's transformer-based architecture specifically designed for multi-horizon forecasting. The state-of-the-art for this exact problem.

**Key innovation:** Separates inputs into three types:
- **Static covariates:** Things that don't change (store location, product category)
- **Known future inputs:** Things you KNOW ahead of time (planned promotions, holidays, day of week)
- **Observed past inputs:** Things you only know historically (past sales, past weather)

This is exactly the promotional forecasting setup — you KNOW the promo is coming in 2 weeks.

**Architecture components:**
- Variable selection networks (automatically learns which features matter)
- Gated residual networks (handles different types of processing)
- Multi-head attention (captures long-range dependencies)
- Quantile regression outputs (probabilistic forecasts)

**Strengths:**
- State-of-the-art accuracy for multi-horizon forecasting
- Handles known future inputs natively (perfect for promotions)
- Built-in attention-based interpretability (which features drove the prediction?)
- Probabilistic output (quantile forecasts)
- Single model for all series (scalable)

**Weaknesses:**
- Complex architecture — harder to implement and debug
- Needs substantial data (100K+ samples across series)
- Computationally expensive to train
- Relatively new — less battle-tested in production than XGBoost

**When to use for Akuret:** The aspirational target. If data volume and engineering resources support it, TFT is the best-fit architecture for promotional demand forecasting. Start with XGBoost, graduate to TFT.

**Python (PyTorch Forecasting library):**
```python
from pytorch_forecasting import TemporalFusionTransformer

model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
)
trainer.fit(model, train_dataloaders=train_dataloader)
predictions = model.predict(test_dataloader)
```

---

## 9. Model Comparison Matrix

| Model | Type | Handles Promos | Multi-series | Probabilistic | Data Needed | Speed | Interpretable |
|---|---|---|---|---|---|---|---|
| **ARIMA/SARIMA** | Statistical | No (univariate) | No (1 per series) | Limited | Small | Fast | High |
| **ETS/Holt-Winters** | Statistical | No | No | No | Small | Very fast | High |
| **Prophet** | Statistical | Yes (regressors) | No | Yes | Small-Med | Medium | High |
| **XGBoost/LightGBM** | ML | Yes (features) | Yes (global) | No* | Medium | Fast | Medium |
| **Random Forest** | ML | Yes (features) | Yes | No | Medium | Medium | Medium |
| **LSTM/GRU** | Deep Learning | Yes | Yes | No* | Large | Slow | Low |
| **N-BEATS** | Deep Learning | No (univariate) | Yes | No | Large | Medium | Medium |
| **DeepAR** | Deep Learning | Yes | Yes (global) | Yes | Large | Slow | Low |
| **TFT** | Deep Learning | Yes (native!) | Yes (global) | Yes | Very Large | Slow | Medium |

*Can be made probabilistic with quantile regression variants

### Recommended progression for Akuret

```
Phase 1 (Week 1-2):     Prophet + XGBoost baseline
Phase 2 (Week 3-4):     Feature-rich LightGBM with tuning
Phase 3 (Month 2):      DeepAR on SageMaker (probabilistic)
Phase 4 (Month 3+):     TFT if data volume justifies it
```

---

## 10. Clustering as Preprocessing

### Why clustering before forecasting?

Not all stores respond to promotions the same way. Not all products have the same lift patterns. Clustering groups similar entities so we can:
- Train separate models per cluster (more accurate)
- Use cluster membership as a feature in a global model
- Handle cold-start (new store → assign to cluster → use cluster model)

### Store clustering

**Features to cluster on:**
- Average weekly revenue
- Store size (sq ft)
- Location type (urban, suburban, rural)
- Customer demographics
- Historical promotional response rate

**Algorithms:**
- **K-Means** — simple, fast, good for well-separated clusters
- **DBSCAN** — handles arbitrary shapes, identifies outlier stores
- **Hierarchical** — gives a dendrogram showing cluster relationships

**Example result:**
```
Cluster A: "High-traffic urban" — 120 stores, avg lift 2.8x
Cluster B: "Suburban family"    — 250 stores, avg lift 2.1x
Cluster C: "Rural low-traffic"  — 80 stores, avg lift 1.4x
```

### Product clustering

**Features to cluster on:**
- Price point, category, brand
- Baseline velocity (units/week)
- Promotional elasticity (how much does a 10% discount increase sales?)
- Substitutability (Coke/Pepsi are substitutes; Coke/chips are complements)

### Customer segmentation (if customer-level data exists)

**RFM Analysis:**
- **Recency** — how recently they purchased
- **Frequency** — how often they purchase
- **Monetary** — how much they spend

This helps answer: "Which customer segments respond most to promotions?"

---

## 11. Evaluation Metrics for Forecasting

### Primary metrics (NOT classification metrics)

| Metric | Formula | What it measures | When to use |
|---|---|---|---|
| **MAE** | mean(\|actual - predicted\|) | Average error in units | General accuracy |
| **MAPE** | mean(\|actual - predicted\| / actual) × 100 | Average % error | Comparing across products |
| **RMSE** | sqrt(mean((actual - predicted)²)) | Penalizes large errors | When big misses are costly |
| **WAPE** | sum(\|actual - predicted\|) / sum(actual) | Weighted % error | Better than MAPE for low-volume items |
| **Forecast Bias** | mean(predicted - actual) / mean(actual) | Systematic over/under | Critical for inventory |
| **Coverage** | % of actuals within prediction interval | Calibration of uncertainty | For probabilistic models |

### Why bias matters more than accuracy for Akuret

```
Scenario A: MAPE = 15%, Bias = +5% (slightly over-predicting)
  → Order a bit too much → small waste, but shelves are stocked
  
Scenario B: MAPE = 12%, Bias = -10% (under-predicting)
  → Order too little → stockouts during promo → lost sales + angry customers

Scenario B has "better accuracy" but WORSE business outcome.
```

**For grocery retail, optimize for slightly positive bias during promotions** — over-ordering is cheaper than stockouts.

### Backtesting (walk-forward validation)

Never use random train/test split for time series — it leaks future data!

```
CORRECT: Walk-forward validation

|---Train (Week 1-50)---|--Test (Week 51-52)--|
|---Train (Week 1-52)---|--Test (Week 53-54)--|
|---Train (Week 1-54)---|--Test (Week 55-56)--|

Each fold trains on past, tests on future. Average metrics across folds.

WRONG: Random 80/20 split
  → Model sees March 2024 in training but predicts February 2024 in test
  → Artificially inflated accuracy (data leakage)
```

---

## 12. The Agent Architecture

### What the agent does

The agent automates the entire weekly forecasting cycle — no human intervention for standard predictions, human review only for edge cases.

### Agent workflow

```
TRIGGER: New promotion announced (or weekly schedule)
    │
    ▼
STEP 1: Data Collection
    ├── Pull transactional data from POS/ERP
    ├── Pull promotion calendar (next 2 weeks)
    ├── Pull external data (holidays, weather forecast)
    ├── Pull inventory levels (current stock)
    │
    ▼
STEP 2: Preprocessing
    ├── Clean data (handle missing values, outliers)
    ├── Engineer features (lags, rolling stats, promo encoding)
    ├── Assign store/product clusters
    │
    ▼
STEP 3: Forecasting
    ├── Run baseline model (Prophet/ETS — "what would sell without promo")
    ├── Run promotional model (XGBoost/TFT — "what will sell with promo")
    ├── Calculate lift: promotional - baseline
    ├── Generate confidence intervals
    │
    ▼
STEP 4: Decision Support
    ├── Calculate recommended order quantities
    ├── Flag high-uncertainty predictions for human review
    ├── Detect cannibalization risks
    ├── Estimate revenue impact
    │
    ▼
STEP 5: Action
    ├── Push order recommendations to inventory system
    ├── Generate task list for store managers
    ├── Log predictions for future model evaluation
    │
    ▼
STEP 6: Feedback Loop
    ├── After promo ends: compare predicted vs actual
    ├── Update model performance metrics
    ├── Trigger retraining if accuracy degrades
    └── Store results for future promo planning
```

### LLM layer — where it fits

The LLM does NOT do the forecasting. The LLM provides:
- **Natural language interface** — "What's the predicted lift for the BOGO on Coke next week in our downtown stores?"
- **Report generation** — converting model output into readable summaries for managers
- **Anomaly explanation** — "Why is the predicted lift unusually high?" → LLM synthesizes from feature importance
- **Orchestration** — the LLM-based agent coordinates the pipeline steps, handles errors, and routes decisions

```
Forecasting: Classical/ML models (XGBoost, Prophet, DeepAR)
Orchestration: LLM-based agent (LangChain/LangGraph)
Interface: Natural language queries and reports
```

---

## 13. Complete Pipeline — End to End

### Technical architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  POS/ERP ──→ ETL Pipeline ──→ Feature Store ──→ Time Series DB  │
│  Weather API ──┘                                                 │
│  Promo Calendar ──┘                                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODELING LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Baseline    │  │  Promotional │  │    Ensemble /      │    │
│  │   (Prophet)   │  │  (XGBoost)   │  │    Reconciliation  │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   DeepAR     │  │     TFT      │  ← Future upgrades         │
│  │  (SageMaker) │  │  (PyTorch)   │                             │
│  └──────────────┘  └──────────────┘                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER                                 │
│  LangChain/LangGraph Agent                                       │
│  ├── Orchestrates weekly pipeline                                │
│  ├── Routes predictions to stakeholders                          │
│  ├── Flags anomalies for human review                            │
│  ├── Generates natural language reports                          │
│  └── Monitors model performance                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ACTION LAYER                                │
│  Order Recommendations ──→ Inventory System                      │
│  Task Lists ──→ Store Managers                                   │
│  Dashboards ──→ Category Managers / Executives                   │
│  Performance Reports ──→ Feedback Loop                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14. Time Series Databases & Tools

### Databases

| Database | Type | Best for | Notes |
|---|---|---|---|
| **TimescaleDB** | PostgreSQL extension | Time series on top of Postgres | You already know Postgres! |
| **InfluxDB** | Purpose-built TSDB | High-volume metrics/IoT | Popular, good query language (Flux) |
| **Apache Druid** | Columnar analytics | Real-time analytics on event data | Good for dashboards |
| **Amazon Timestream** | Managed TSDB on AWS | AWS-native time series | Integrates with SageMaker |
| **QuestDB** | High-performance TSDB | Fast ingestion, SQL interface | Good for real-time |
| **ClickHouse** | Columnar OLAP | Fast analytical queries on large datasets | Good for aggregations |

**Recommendation for Akuret:** TimescaleDB (Postgres extension — fits your existing stack) or Amazon Timestream (AWS-native, integrates with SageMaker).

### ML/Forecasting libraries

| Library | Language | Models | Notes |
|---|---|---|---|
| **statsmodels** | Python | ARIMA, SARIMA, ETS, VAR | Classical statistics |
| **Prophet** | Python/R | Prophet | Facebook/Meta, easy to use |
| **scikit-learn** | Python | RF, SVR, basic ML | Feature engineering toolkit |
| **XGBoost** | Python/R | XGBoost | Gradient boosting |
| **LightGBM** | Python/R | LightGBM | Faster than XGBoost, handles categoricals |
| **PyTorch Forecasting** | Python | TFT, N-BEATS, DeepAR | Deep learning forecasting |
| **GluonTS** | Python | DeepAR, WaveNet, Transformer | Amazon's forecasting library |
| **Darts** | Python | ALL OF THE ABOVE | Unified API for many models |
| **NeuralProphet** | Python | Prophet + Neural Network | Prophet with deep learning extensions |
| **sktime** | Python | Many | Scikit-learn compatible time series |
| **tsfresh** | Python | Feature extraction | Automatic time series feature engineering |

### Feature stores

| Tool | What it does |
|---|---|
| **Feast** | Open-source feature store — serves features for training and inference |
| **Amazon SageMaker Feature Store** | Managed feature store on AWS |
| **Tecton** | Enterprise feature platform |

### Orchestration / Agent tools

| Tool | What it does |
|---|---|
| **LangChain** | Build LLM-powered agents and chains |
| **LangGraph** | Stateful, multi-step agent workflows with checkpointing |
| **Airflow** | Schedule and monitor data pipelines (DAGs) |
| **Prefect** | Modern alternative to Airflow |
| **Celery + Redis** | Async task processing (you already use this!) |

---

## 15. Key Concepts You Must Know

### Autocorrelation

The correlation of a time series with a lagged version of itself. If today's sales are strongly correlated with last Tuesday's sales, there's weekly autocorrelation. Plotted in an **ACF (Autocorrelation Function)** chart.

**Why it matters:** Helps determine the `p` and `q` parameters for ARIMA, and tells you which lag features to engineer.

### Stationarity (revisited)

Most models assume the data's statistical properties don't change over time. Test with **ADF (Augmented Dickey-Fuller)** test. Make non-stationary data stationary via differencing or transformation.

### Exogenous variables

External factors that influence the forecast but are not part of the time series itself. For Akuret: promotions, holidays, weather, competitor actions. Models that handle these: SARIMAX (SARIMA with eXogenous variables), Prophet (regressors), XGBoost (features), TFT (known future inputs).

### Promotional elasticity

How sensitive demand is to the discount depth.

```
Elasticity = % change in quantity / % change in price

Elastic (|e| > 1):   10% discount → 20% sales increase (beverages, snacks)
Inelastic (|e| < 1): 10% discount → 5% sales increase (bread, milk)
```

**Why it matters:** High-elasticity products benefit more from promotions. This is a key feature for the model.

### Cannibalization and halo effects

- **Cannibalization:** Coke promo steals sales from Pepsi (substitute products)
- **Halo effect:** Coke promo increases chip sales (complementary products)
- **Forward buying:** Customers stockpile during promo → post-promo sales dip

All three must be modeled to avoid over- or under-ordering related products.

### Forecast reconciliation

When forecasts exist at multiple hierarchy levels (product → category → department → store → region → national), they must be consistent — store-level forecasts should sum to the regional total.

**Methods:**
- **Bottom-up:** Forecast at SKU/store level, aggregate up
- **Top-down:** Forecast at national level, distribute down by historical proportions
- **Middle-out:** Forecast at an intermediate level, reconcile both directions
- **Optimal reconciliation (MinT):** Statistical method that finds the best combination

---

## 16. Common Pitfalls in Demand Forecasting

| Pitfall | What goes wrong | How to avoid |
|---|---|---|
| **Using classification metrics** | Accuracy/recall are for classifiers, not forecasters | Use MAE, MAPE, RMSE, bias |
| **Random train/test split** | Leaks future data into training | Use walk-forward validation |
| **Ignoring stockout bias** | Treating zero-sales days as zero demand | Flag stockout periods, impute true demand |
| **Overfitting to promotion history** | Model memorizes past promos instead of learning patterns | Cross-validate across different promo types |
| **Ignoring cannibalization** | Forecast each product independently | Model cross-product effects |
| **Point forecasts only** | "We'll sell 250 units" gives no decision boundary | Use probabilistic models (DeepAR, TFT) |
| **Same model for all products** | High-volume Coke needs different model than niche organic juice | Cluster products, separate models or features |
| **Not monitoring in production** | Model degrades silently | Track MAPE, bias, data drift weekly |

---

## 17. Recommended Approach for Akuret

### Phase 1: Understand & Baseline (Week 1-2)

1. **Data exploration** — understand the transactional data, identify quality issues
2. **Decomposition** — decompose top SKUs into trend + seasonality + residual
3. **Baseline models** — run Prophet and ETS on top 50 SKUs
4. **Establish metrics** — set MAE, MAPE, bias benchmarks

### Phase 2: Feature Engineering & ML (Week 3-4)

1. **Build feature pipeline** — lag features, rolling stats, promo encoding, calendar features
2. **Store/product clustering** — segment by behavior
3. **Train XGBoost/LightGBM** — feature-rich model with promo inputs
4. **Compare vs baseline** — does ML beat Prophet? By how much?
5. **Feature importance analysis** — which features drive accuracy?

### Phase 3: Production & Agent (Month 2)

1. **Deploy best model** — FastAPI serving endpoint, Docker, CI/CD
2. **Build agent pipeline** — LangGraph workflow for weekly automation
3. **Monitoring dashboard** — track MAPE, bias, data drift
4. **Human-in-the-loop** — flag high-uncertainty predictions for review

### Phase 4: Advanced Models (Month 3+)

1. **DeepAR on SageMaker** — probabilistic forecasts with confidence intervals
2. **TFT evaluation** — if data volume supports it
3. **Cannibalization modeling** — cross-product effects
4. **Continuous improvement** — feedback loop, retraining triggers

---

## 18. Interview Q&A — Forecasting

### "How would you approach the promotional forecasting problem?"

> "I'd start by understanding the data — the transactional history, promotion calendar, and any external signals like holidays and seasonality. Then I'd establish a baseline with something like Prophet, which handles seasonality well and lets me add promotions as regressors. From there, I'd build a feature-rich model using XGBoost or LightGBM, engineering lag features, rolling statistics, and promo-specific features like discount depth, promo type, and historical lift for similar products. The key insight is that feature engineering matters more than model complexity for this type of problem. Once we have a model that beats the baseline, we wrap it in an agent that automates the weekly cycle — data pull, feature computation, forecast, order recommendation, and monitoring."

### "How do you evaluate a forecasting model?"

> "Walk-forward validation, never random splits — train on the past, test on the future, slide forward. Primary metrics are MAPE for overall accuracy and forecast bias for operational impact. Bias matters more than raw accuracy for inventory — under-predicting during a promo means stockouts and lost sales, which is more expensive than slightly over-ordering. I'd also track coverage if using probabilistic models — what percentage of actual values fall within the prediction interval."

### "Would you use a pretrained model or train from scratch?"

> "It depends on the data. If the transactional data is substantial — say, a year or more of weekly data across many SKUs and stores — I'd train from scratch using XGBoost with engineered features, because the model needs to learn this specific retailer's patterns. For deep learning approaches like DeepAR, there are pretrained weights that can provide a warm start, but fine-tuning on the actual data is still necessary. The key question I need to answer after seeing the data is: do we have enough history to learn promotional patterns reliably, or do we need to bootstrap with transfer learning from similar retail datasets?"

### "Why not use an LLM for the forecasting itself?"

> "LLMs are excellent at language, reasoning, and orchestration, but they're not designed for numerical prediction. They can't reliably predict that Coke sales will be 247 units next Tuesday. That requires statistical or ML models trained on numerical patterns. Where the LLM adds value is in the agent layer — orchestrating the pipeline, generating natural language reports from model output, explaining anomalies, and providing a conversational interface for managers to query forecasts. Classical models for prediction, LLM for orchestration and communication."

### "How do you handle the 1-2 week advance notice constraint?"

> "This actually works in our favor. The promotion schedule is a **known future input** — we know what promotion will run, when, and at what discount depth. Models like Prophet (regressors), XGBoost (features), and TFT (known future inputs) can all incorporate this directly. The forecast becomes: given that we KNOW a BOGO promotion is happening on Coke in 2 weeks, what will the demand be? The 1-2 week window is the prediction horizon — standard for most forecasting models."

### "What about the agent architecture you mentioned?"

> "The agent automates the entire cycle: trigger on new promotion announcement or weekly schedule, pull data from POS and promo calendar, compute features, run the forecast model, generate order recommendations with confidence intervals, push to the inventory system, and monitor actual vs predicted after the promo ends. I'd build this with LangGraph for the stateful workflow and Celery for async task processing — tools I've used in production. The key principle is: the model does the math, the agent does the logistics, humans only review edge cases."

### "How does clustering fit in?"

> "Clustering is a preprocessing step, not the core model. I'd cluster stores by behavior — high-traffic urban stores respond differently to promotions than rural stores. Then I either train separate forecasting models per cluster, or use cluster membership as a feature in a global model. Similarly, I'd cluster products by promotional elasticity — products that respond similarly to discounts can share model parameters, which helps when a specific product has limited promotional history."

---

> **Key correction from first interview:** This is a **forecasting/regression** problem, not classification. The output is a predicted number (units to sell), not a category label. Clustering supports the forecasting but is not the core task.

---

> **Last updated:** August 2026 | **Prepared for:** Akuret Solutions — CTO Interview Follow-up
