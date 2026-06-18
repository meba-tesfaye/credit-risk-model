## 📊 Business Case & Objective

### 1. Business Overview
The goal of this project is to build a robust, data-driven credit risk assessment model for a financial institution. By analyzing historical transaction and borrower data, the model aims to accurately predict the likelihood of default, enabling the business to minimize non-performing loans (NPLs) while optimizing credit approval rates.

### 2. The Proxy Target (RFM-Based Logic)
Because traditional, explicit credit default labels were unavailable in the raw dataset, an engineering proxy target was developed based on **Recency, Frequency, and Monetary (RFM)** behavior:
* **Recency:** Measures the elapsed time since the borrower's last transaction or payment.
* **Frequency:** Evaluates the volume and consistency of account interactions.
* **Monetary:** Tracks the total financial throughput and the key **Transaction-to-Monetary Ratio**.

Borrowers exhibiting high recency (long gaps since last activity), low transaction frequency, and an uncharacteristic drop in their transaction-to-monetary ratio are mathematically flagged as high risk (Proxy Default = 1). This proxy serves as our foundational training target for credit default risk estimation.

---

## 🔍 Key EDA Insights & Analytical Findings

During the Exploratory Data Analysis (EDA) phase, several critical data properties and structural anomalies were uncovered and systematically addressed in the pipeline:

* **Class Imbalance:** The engineering proxy target revealed a heavily skewed distribution between low-risk and high-risk borrowers. To prevent the models from biasing toward the majority class, class-weight configurations and stratified splitting were integrated into the training phase.
* **Feature Skewness:** Financial metrics (such as transaction amounts and account balances) exhibited heavy-tailed distributions. Log-transformations and robust scaling were implemented within `src/data_processing.py` to handle extreme values and normalize feature variances.
* **Feature Correlations:** Multicollinearity checks identified strong correlations among transaction volume metrics. Strategic feature engineering—such as consolidating tracking metrics into structural interaction ratios—was used to reduce redundancy and enhance model stability.
