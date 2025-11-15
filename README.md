# South African Crop Yield Consistency Modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive agricultural analytics project that develops a **Yield Consistency (YC) scoring system** for pre-credit assessment in South African agriculture. This project analyzes historical SAFEX (South African Futures Exchange) and StatsSA data to predict crop yields and quantify agricultural production risk for financial institutions.

[View Repository →](https://github.com/supermarmar/crop_modelling)

## 🎯 Project Overview

The core objective is to create a data-driven credit scoring model that evaluates a farmer's likelihood of achieving their projected crop yields. The system generates a **Yield Consistency Score (0-35 points)** that combines:

- **Regional Yield Benchmarks**: Statistical analysis of 34+ years of provincial crop data
- **Time Series Forecasting**: ARIMA-GARCH models for yield prediction and volatility estimation  
- **Trend Analysis**: Multi-temporal regression analysis of yield patterns
- **Risk Assessment**: Comparative analysis against dry-land production estimates

## 📊 Key Features & Methodology

### Crop Coverage

The system analyzes five major South African crops:

- **White Maize** & **Yellow Maize** (Staple grains)
- **Soybean** (Protein crop)
- **Sunflower** (Oilseed crop)  
- **Wheat** (Winter crop)

### Yield Consistency Scoring Components

#### 1. **Initial Yield Estimate Assessment (25 points)**

Compares farmer's projected yields against:

- Multi-model forecasts (ARIMA, ARIMA-GARCH)
- Historical regional averages and weighted averages
- Dry-land production benchmarks by crop and province

> **Example:** A farmer in Eastern Cape projecting 7.21 t/ha white maize yield receives points based on deviation from regional median of ~4.5 t/ha

#### 2. **Industry Trend Analysis (5 points)**

Evaluates long-term (34-year) vs short-term (5-year) yield trends:

- **Positive trends** (>0.25 t/ha/year): Higher scores
- **Declining trends** (<-0.25 t/ha/year): Lower scores
- **Stable trends**: Moderate scores

#### 3. **Farm-Level Historical Performance (5 points)**

When available, analyzes farmer's historical yield data:

- Trend analysis of past performance
- Consistency metrics
- Comparison with regional benchmarks

### Advanced Time Series Modeling

The project implements sophisticated econometric models:

```python
# Example from yield_consistency_eda.ipynb
def arima_forecast(prov, crop, years, training_size, forecast_len):
    """
    Fits ARIMA model with GARCH/EGARCH volatility modeling
    for robust yield forecasting with uncertainty quantification
    """
    model = auto_arima(train_data, seasonal=False, stepwise=True)
    
    # Volatility modeling on residuals
    garch_models = ["GARCH", "EGARCH", "TGARCH"] 
    best_model = select_best_volatility_model(residuals)
    
    return arima_forecast, combined_forecast
```

### Statistical Testing Framework

- **Stationarity Testing**: Augmented Dickey-Fuller tests for time series properties
- **Model Validation**: Ljung-Box tests for residual autocorrelation
- **Volatility Modeling**: ARCH-family models for risk assessment
- **Forecast Accuracy**: MAE, RMSE, MAPE metrics for model evaluation

## 🗂️ Data Pipeline Architecture

### 01_data_engineering/

**Raw data ingestion and standardization**

- `area_engineering.ipynb`: [Processes SAFEX area data](notebooks/01_data_engineering/area_engineering.ipynb) across provinces and crops
- `delivery_engineering.ipynb`: Cleans and transforms delivery/sales data
- `production_engineering.ipynb`: Standardizes production tonnage data

### 02_eda/

**Exploratory analysis and yield consistency modeling**

- `yield_consistency_eda.ipynb`: [Core YC algorithm development](notebooks/02_eda/yield_consistency_eda.ipynb) with statistical modeling
- `yc_industry_analysis.ipynb`: Industry benchmark analysis and trend evaluation

### 03_feature_engineering/

**Province and crop-specific modeling**

Each crop has dedicated analysis across three domains:

#### Area Analysis

- [`white_maize_area.ipynb`](notebooks/03_feature_engineering/area/white_maize_area.ipynb): Provincial area allocation modeling with rolling averages
- Similar notebooks for yellow maize, soya, sunflower with configuration files

#### Production Forecasting

- Time series models for production volume prediction
- Multi-year averaging techniques for smoothing volatility

#### Delivery Pattern Analysis

- Market delivery trend analysis
- Supply chain timing patterns

> **Example output from white maize area analysis shows Free State dominating with ~60% of national area**

## 🔬 Technical Implementation

### Core Functions

The project implements a comprehensive yield assessment framework:

```python
# From yield_consistency_eda.ipynb - Main scoring function
def yield_consistancy(prov, crop, hist_yield, initial_est, years, training_size):
    """
    Comprehensive yield consistency scoring combining:
    - Regional benchmarking
    - Time series forecasting  
    - Trend analysis
    - Risk bounds checking
    """
    # Multi-model forecasting
    arima_forecast, garch_forecast = arima_forecast(prov, crop, years, 0.85, 7)
    
    # Statistical bounds checking  
    min_yield = min(forecasts + [dry_land_min])
    max_yield = max(forecasts + [dry_land_max])
    
    # YC scoring logic with business rules
    yc_score = calculate_score_components(initial_est, trends, historical)
    
    return yc_score  # 0-35 point scale
```

### Data Sources & Integration

- **SAFEX Historical Data**: 34+ years of crop area, production, delivery data
- **Provincial Coverage**: All 9 South African provinces with crop-specific analysis
- **Temporal Resolution**: Annual data with weekly resampling capabilities
- **Validation Data**: Cross-validation using 85/15 train-test splits

## 🎯 Business Application

### Credit Risk Assessment

The YC score provides quantified risk metrics for agricultural lending:

- **Low Risk (25-35 points)**: Realistic projections with strong historical backing
- **Medium Risk (15-24 points)**: Moderate deviations requiring closer monitoring  
- **High Risk (0-14 points)**: Unrealistic projections requiring yield insurance or collateral

### Example Use Case

```text
Farmer Profile: Eastern Cape, White Maize, 7.21 t/ha projection
- YC Score: 18/35 points
- Regional median: 4.5 t/ha  
- Risk assessment: Moderately optimistic projection
- Recommendation: Request yield insurance or additional collateral
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources (SAFEX, StatsSA)
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│       ├── safex/     <- SAFEX crop delivery, estimate, and financial data
│       └── stats_sa/  <- Statistics South Africa agricultural data
│
├── docs               <- Project documentation
├── models             <- Trained and serialized models, model predictions, summaries
├── notebooks          <- Jupyter notebooks with structured analysis workflow:
│   ├── 01_data_engineering/    <- Data ingestion, cleaning, and standardization
│   ├── 02_eda/                 <- Exploratory analysis and YC algorithm development  
│   └── 03_feature_engineering/ <- Crop and province-specific modeling
│       ├── area/               <- Area allocation models by crop
│       ├── delivery/           <- Delivery pattern analysis
│       └── production/         <- Production forecasting models
│
├── references         <- Data dictionaries, manuals, and explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures for reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── pyproject.toml     <- Project configuration file with package metadata
├── setup.cfg          <- Configuration file for flake8
│
└── src/              <- Source code for use in this project
    ├── __init__.py           <- Makes src a Python module
    ├── config.py             <- Store useful variables and configuration
    ├── dataset.py            <- Scripts to download or generate data
    ├── features.py           <- Code to create features for modeling
    ├── io_data_model.py      <- Data loading, saving, and model I/O utilities
    ├── transformation.py     <- Data cleaning and preprocessing functions
    ├── time_series.py        <- Time series analysis utilities
    ├── anomaly.py            <- Anomaly detection functions
    ├── plots.py              <- Code to create visualizations
    └── modeling/             <- Machine learning and statistical modeling
        ├── predict.py        <- Code to run model inference
        └── train.py          <- Code to train models
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scipy, scikit-learn, statsmodels, pmdarima, arch
- Jupyter Notebook or JupyterLab for running analysis notebooks

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd crop_modelling

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
```

### Usage

1. **Data Engineering**: Start with notebooks in `01_data_engineering/` to process raw SAFEX data
2. **Exploratory Analysis**: Run `02_eda/yield_consistency_eda.ipynb` to understand the YC methodology
3. **Feature Engineering**: Use crop-specific notebooks in `03_feature_engineering/` for detailed analysis
4. **Scoring**: Apply the yield consistency function to evaluate farmer projections

```python
# Example usage of the main scoring function
from notebooks.eda.yield_consistency_eda import yield_consistancy

# Score a farmer's yield projection
yc_score = yield_consistancy(
    prov="Free State",
    crop="white maize", 
    hist_yield=[4.2, 4.5, 3.8, 4.1, 4.6],  # Farmer's historical yields
    initial_est=5.2,  # Farmer's projection for upcoming season
    years=34,         # Historical data depth
    training_size=0.85  # Train/test split
)
```

## 📈 Results & Impact

- **34+ years** of South African crop data analyzed
- **9 provinces** covered with crop-specific models
- **5 major crops** modeled (maize, soya, sunflower, wheat)
- **Quantified risk assessment** for agricultural lending
- **Multi-model approach** combining statistical and machine learning techniques

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SAFEX (South African Futures Exchange) for providing historical crop data
- Statistics South Africa (StatsSA) for agricultural statistics
- Agricultural industry experts for domain knowledge and validation

--------

