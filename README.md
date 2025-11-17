# shipping-delay-prediction-ml
End-to-end ML project to predict shipment delays using logistic regression, random forest, and XGBoost.
# Predicting Shipment Delays Using Machine Learning  
**Author:** Mohamed Sahidu Bah  

This project applies machine learning to predict **shipment delays** using a logistics dataset of 2,000 shipments.  
It combines:

- Business-driven feature engineering  
- Exploratory Data Analysis (EDA)  
- Logistic Regression, Random Forest, and XGBoost models  
- Model comparison and feature importance analysis  

The goal is to help logistics teams **identify high-risk shipments before they are delayed** and act proactively.

---

## 1. Business Context

Late deliveries increase cost, damage customer trust, and disrupt planning.  
Being able to flag shipments that are **likely to be delayed** lets a logistics company:

- Reroute or prioritize risky shipments  
- Engage with underperforming carriers  
- Inform customers earlier  
- Optimize resources around peak periods  

This project simulates that workflow using historical shipment data.

---

## 2. Dataset

**Rows:** 2,000 shipments  
**Columns (simplified):**

- `Shipment_ID` – unique shipment identifier  
- `Origin_Warehouse` – origin hub (e.g., Warehouse_MIA, Warehouse_BOS)  
- `Destination` – city-level destination (e.g., San Francisco, Boston)  
- `Carrier` – shipping provider (UPS, DHL, FedEx, Amazon Logistics, etc.)  
- `Shipment_Date`, `Delivery_Date` – departure and delivery timestamps  
- `Transit_Days` – planned transit duration  
- `Weight_kg` – shipment weight  
- `Cost` – shipping cost  
- `Distance_miles` – route distance  
- `Status` – Delivered, Delayed, Lost, Returned, In Transit  

### Target Variable: `is_delayed`

`Status` is mapped into a binary target:

- **1 (delayed)** → Delayed, Lost, Returned  
- **0 (on time)** → Delivered, In Transit  

Class balance:

- 0 → 86.2%  
- 1 → 13.8%  

So the task is **imbalanced classification**.

---

## 3. Data Cleaning

Key steps:

- Converted `Shipment_Date` and `Delivery_Date` to datetime.  
- Reconstructed 32 missing `Delivery_Date` values using:  
  \> `Delivery_Date = Shipment_Date + Transit_Days`  
- Imputed 41 missing `Cost` values with the **median** (robust to outliers).  
- Confirmed there were **no duplicate rows**.

---

## 4. Feature Engineering

Business-focused engineered features:

- `actual_transit_days`  
  - `(Delivery_Date - Shipment_Date).days`  
- `planned_vs_actual_diff`  
  - `actual_transit_days - Transit_Days`  
  - Positive → delivery took longer than planned  
  - Negative → delivery was faster than planned  
- `shipment_month`  
  - Captures seasonal effects (e.g., end-of-year peak)  
- `shipment_dayofweek` (0 = Monday … 6 = Sunday)  
- `is_weekend_shipment` (1 if Saturday/Sunday, else 0)  
- `route` = `Origin_Warehouse` + "_" + `Destination`  
  - Captures route-level risk  
- `Weight_kg_log` = `log1p(Weight_kg)`  
- `Cost_log` = `log1p(Cost)`  

These features reflect **how operations actually work**: timing, geography, carriers, and route behavior.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Continuous variables

- `Distance_miles` is roughly symmetric (skew ≈ 0.04) with no extreme outliers.  
- `Transit_Days` is slightly right-skewed (skew ≈ 0.40).  
- Raw `Weight_kg` and `Cost` are **heavily right-skewed** with large outliers.  
- After log-transform:
  - `Weight_kg_log` becomes nearly symmetric (skew ≈ 0.28)  
  - `Cost_log` has mild skew (skew ≈ -0.60)  

`actual_transit_days` and `planned_vs_actual_diff` are **strongly left-skewed**  
(e.g., skew ≈ -3.30 and -4.52), which is expected: most shipments cluster near the planned time, while a minority experience significant delays.

### 5.2 Categorical delay rates

Delay rate by origin warehouse, destination, and carrier was calculated using bar plots:

- Certain **warehouses** (e.g., Warehouse_SF, Warehouse_HOU) show higher delay rates.  
- Some **destinations** (e.g., Boston, Detroit) have noticeably higher delay rates than others.  
- Among **carriers**, Amazon Logistics and LaserShip appear riskier than UPS or USPS.

These patterns are consistent with the feature importances found later.

---

## 6. Modeling Approach

Three models were trained:

1. **Logistic Regression** (baseline, interpretable)  
2. **Random Forest** (non-linear, robust)  
3. **XGBoost** (gradient boosting, best performance)

### 6.1 Encoding and splits

- Categorical variables encoded using **one-hot encoding** for:
  - `Origin_Warehouse`, `Destination`, `Carrier`, `route`
- Data split:
  - 80% train, 20% test  
  - `train_test_split(..., stratify=y, random_state=42)`  
- For Logistic Regression:
  - Dropped multicollinear raw variables:
    - `Weight_kg`, `Cost`, `Transit_Days`, `actual_transit_days`
  - Kept:
    - `Weight_kg_log`, `Cost_log`, `planned_vs_actual_diff`, `Distance_miles`, `shipment_month`, `shipment_dayofweek`, `is_weekend_shipment`
  - Standardized numeric features using `StandardScaler`.
  - Used `class_weight="balanced"` to handle class imbalance.

---

## 7. Model Results

### 7.1 Logistic Regression (baseline)

- **ROC-AUC:** ~**0.74**  
- **Accuracy:** ~**0.885**  
- **Delayed class (1):**
  - Precision: **0.58**  
  - Recall: **0.56**  
  - F1-score: **0.57**  

Logistic Regression performs reasonably well and is easy to interpret, but it struggles to fully capture the non-linear relationships in the data.

---

### 7.2 Random Forest

- **ROC-AUC:** ~**0.76**  
- **Accuracy:** ~**0.968**  
- **Delayed class (1):**
  - Precision: **1.00**  
  - Recall: **0.56**  
  - F1-score: **0.72**  

**Confusion Matrix (RF):**

- True On-time (0) correctly identified: 345  
- True Delayed (1) correctly identified: 31  
- Missed delays (false negatives): 24  

Random Forest significantly improves overall performance and F1 for the delayed class while maintaining very high precision.

---

### 7.3 XGBoost (best model)

- **ROC-AUC:** ~**0.80**  
- **Accuracy:** ~**0.94**  
- **Delayed class (1):**
  - Precision: **0.97**  
  - Recall: **0.58**  
  - F1-score: **0.73**  

**Confusion Matrix (XGBoost):**

- True On-time (0) correctly identified: 344  
- True Delayed (1) correctly identified: 32  
- Missed delays: 23  

XGBoost achieves the **highest ROC-AUC and the strongest F1-score for delayed shipments** while keeping false positives low.

---

## 8. Feature Importance Insights

### 8.1 Random Forest – Top Features

Top RF features include:

1. `planned_vs_actual_diff`  
2. `actual_transit_days`  
3. `Cost_log` and `Cost`  
4. `Distance_miles`  
5. `Weight_kg_log` and `Weight_kg`  
6. `shipment_month`, `shipment_dayofweek`  
7. `Transit_Days`  
8. Carrier and origin warehouse dummies (e.g., `Carrier_LaserShip`, `Origin_Warehouse_Warehouse_SF`)

### 8.2 XGBoost – Top Features

XGBoost emphasizes:

1. `planned_vs_actual_diff` (by far the strongest)  
2. `actual_transit_days`  
3. Specific high-risk routes, e.g.:
   - `route_Warehouse_MIA_Chicago`  
   - `route_Warehouse_SF_Detroit`  
   - `route_Warehouse_BOS_Portland`  
   - `route_Warehouse_ATL_Minneapolis`  
4. `Transit_Days`  
5. Several carrier and destination dummies (e.g., `Carrier_DHL`, `Destination_Seattle`, `Destination_Houston`, `Carrier_LaserShip`)

**Takeaway:**  
Delays are strongly driven by **how much longer shipments take than planned**, plus route patterns, distance, carrier behavior, and timing (month/day-of-week).

---

## 9. Conclusions

- All three models can predict shipment delays with useful accuracy.  
- **XGBoost** is the best-performing model with:
  - ROC-AUC ≈ **0.80**  
  - Strong precision and recall on the delayed class  
- `planned_vs_actual_diff` and `actual_transit_days` are the dominant predictors, along with:
  - Distance  
  - Specific route combinations  
  - Carrier performance  

This suggests logistics teams should:

- Monitor planned vs actual transit times by route and carrier  
- Focus on underperforming routes (e.g., certain Warehouse–City combinations)  
- Use the model to flag high-risk shipments for proactive intervention.

---

## 10. Future Work

Possible extensions:

- Incorporate **weather data** and **holiday calendars** per route  
- Add features for **shipment priority** or **customer importance**  
- Deploy the XGBoost model as an API (FastAPI / Flask)  
- Build a **Streamlit dashboard** for real-time delay risk monitoring  
- Use **SHAP** values for deeper model explainability at the shipment level  

---

## 11. Project Structure

```bash
shipping-delay-prediction-ml/
│
├── data/
│   └── logistics_shipments_dataset.csv
│
├── notebooks/
│   └── 01_eda.ipynb        # EDA + preprocessing + modeling
│
├── visuals/
│   ├── delay_rate_by_carrier.png
│   ├── delay_rate_by_destination.png
│   ├── delay_rate_by_warehouse.png
│   ├── numeric_distributions.png
│   ├── rf_confusion_matrix.png
│   ├── xgb_confusion_matrix.png
│   └── feature_importances_xgb.png
│
├── README.md
└── requirements.txt
