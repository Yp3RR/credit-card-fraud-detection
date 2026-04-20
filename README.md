# credit-card-fraud-detection

# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset of 284,807 records using multiple classification algorithms, feature engineering, and threshold optimization.

---

## 📌 Problem Statement

Fraud transactions account for only **0.17%** of all transactions (492 out of 284,807). A naive model that predicts everything as legitimate achieves 99.83% accuracy — yet catches zero fraud. The real challenge is **maximizing fraud detection (recall) while maintaining acceptable precision**, making standard accuracy a useless metric here.

---

## 📂 Repository Structure

```
creditcard-fraud-detection/
├── notebooks/
│   └── credit_card_fraud_detection.ipynb   # Main notebook
├── images/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_correlation_with_fraud.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── pr_and_roc_curves.png
│   ├── confusion_matrix_best.png
│   ├── recall_precision_improvement.png
│   └── final_pr_curve.png
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions by European cardholders (September 2013)
- **Features:** V1–V28 (PCA-transformed for confidentiality), `Time`, `Amount`, `Class`
- **Target:** `Class` — 0 = Legitimate, 1 = Fraud
- **Class imbalance:** 0.17% fraud rate

> The CSV file is not included in this repo due to size. Download `creditcard.csv` from Kaggle and place it in the root directory before running the notebook.

---

## 🔧 Approach

### 1. Exploratory Data Analysis
- Verified zero missing values
- Visualized extreme class imbalance
- Analyzed transaction amount distributions across fraud vs legitimate
- Identified top PCA features most correlated with fraud

### 2. Feature Engineering
Created 8 new features from `Time` and `Amount`:

| Feature | Description |
|---|---|
| `Hour` | Hour of day extracted from Time (seconds) |
| `Is_Night` | Binary flag — 1 if transaction between 10pm–5am |
| `Day` | Day number of the transaction |
| `Log_Amount` | Log transform to reduce Amount skewness |
| `Amount_Sq` | Squared amount — amplifies signal for large transactions |
| `Amount_Sqrt` | Square root of amount |
| `V_Magnitude` | L2 norm across all V columns — overall deviation from typical transaction |
| `Amount_Zscore` | Standard deviations away from mean amount |
| `Vn_x_LogAmt` | Interaction terms between top 5 fraud-signal PCA components and Log_Amount |

> Note: V1–V28 are PCA components and are not directly interpretable in business terms. For business narratives, `Amount`, `Hour`, and `Is_Night` are the meaningful features.

### 3. Feature Scaling
Applied `RobustScaler` (resistant to outliers) to Time, Amount, and engineered features. V1–V28 were left unscaled as they are already normalized from PCA.

### 4. Feature Selection
Used Random Forest feature importance on a 50K sample to select the **top 30 most predictive features** for modeling.

### 5. Train/Test Split — Temporal
Sorted transactions chronologically by `Time` and used an **80/20 temporal split** — first 80% for training, last 20% for testing. This avoids look-ahead bias and reflects real deployment conditions where the model must generalize to future transactions.

### 6. Class Imbalance Handling
Compared two strategies using Logistic Regression as a controlled baseline:

| Strategy | Approach | Result |
|---|---|---|
| `class_weight='balanced'` | Increases loss penalty for fraud misclassification during training | Fast, no synthetic data risk |
| SMOTE | Generates synthetic fraud samples by interpolating between existing ones | Comparable performance, higher computation |

**Chosen approach:** `class_weight='balanced'` — safer for production, avoids overfitting to synthetic samples, performs comparably on this dataset.

### 7. Models Trained
All models used `class_weight='balanced'`:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

### 8. Hyperparameter Tuning
`GridSearchCV` with `StratifiedKFold` (5-fold) on Logistic Regression and Random Forest, optimizing for **F1-score** on the fraud class.

### 9. Threshold Optimization
Default classifiers predict fraud if probability ≥ 0.5. By tuning the classification threshold on the best model, we found the lowest threshold where **precision ≥ 85%**, maximizing recall at that operating point.

---

## 📈 Results

### Recall Improvement — Stage by Stage

| Stage | Technique | Recall | Precision |
|---|---|---|---|
| Baseline | Logistic Regression, no class weight | ~61% | High |
| + Class Weight | `class_weight='balanced'`, threshold=0.5 | Improved | Moderate |
| + Threshold Tuning | Best model + optimal threshold | **~89%** | **≥85%** |

**Key insight:** Class weight reshapes what the model learns during training. Threshold tuning adjusts where on the precision-recall curve we operate at inference time. These two techniques are complementary.

### Why PR-AUC over ROC-AUC?
With 0.17% fraud, ROC-AUC is misleadingly high because it includes true negatives (284K legitimate transactions). The **Precision-Recall curve** focuses only on the minority class, giving a more honest picture of fraud detection performance.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn

---

## 🚀 How to Run

1. Clone the repo
2. Download `creditcard.csv` from Kaggle and place it in the root directory
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
4. Open and run `notebooks/credit_card_fraud_detection.ipynb`

---

## 📋 Key Takeaways

- Accuracy is a misleading metric for imbalanced datasets — always use Precision, Recall, F1, and PR-AUC
- Temporal train/test splits are critical for time-series financial data to avoid look-ahead bias
- Class weighting and threshold tuning are complementary tools — one changes the model, the other changes how you use it
- PR-AUC is more informative than ROC-AUC when the positive class is rare
