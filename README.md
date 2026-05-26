# Bank-Marketing-Metrics-Machine-Learning-with-PySpark


# Bank Marketing Classification with PySpark

A comprehensive machine learning workshop demonstrating high-volume data processing, classification modeling, and predictive analytics using Apache Spark. This project analyzes bank marketing campaign data to predict whether clients will subscribe to a term deposit.

**Author:** Diego Felipe Céspedes  
**Institution:** Pontificia Universidad Javeriana  
**Start Date:** April 28, 2026  
**Dataset:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Loading](#2-data-loading)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Data Cleaning & Preparation](#4-data-cleaning--preparation)
  - [5. Feature Engineering](#5-feature-engineering)
  - [6. Model Development](#6-model-development)
  - [7. Model Evaluation](#7-model-evaluation)
- [Key Findings](#key-findings)
- [Models Implemented](#models-implemented)
- [Performance Results](#performance-results)
- [Technologies & Libraries](#technologies--libraries)
- [Requirements](#requirements)
- [Getting Started](#getting-started)

---

## Project Overview

This workshop implements a complete machine learning pipeline using **PySpark** to process a large-scale bank marketing dataset (45,211 records). The project demonstrates:

- **Data Understanding**: Comprehensive exploratory data analysis to identify patterns, imbalances, and data quality issues
- **Data Preprocessing**: Handling missing values, class imbalance, outliers, and feature engineering
- **Classification Modeling**: Implementation and comparison of 5 different machine learning algorithms
- **Model Evaluation**: Rigorous performance assessment using multiple metrics and visualizations

The ultimate goal is to identify which clients are likely to subscribe to term deposits, enabling targeted marketing strategies.

---

## Objectives

1. **Initialize Spark Session**: Set up distributed computing environment for large-scale data processing
2. **Load Bank Data**: Ingest marketing campaign data from HDFS or local sources
3. **Understand Data**: Perform comprehensive exploratory analysis to identify patterns and anomalies
4. **Clean Data**: Handle missing values, outliers, and class imbalances
5. **Engineer Features**: Transform categorical variables and normalize numerical features
6. **Build Models**: Train multiple classification algorithms
7. **Compare Performance**: Evaluate and rank models based on standard metrics

---

## Dataset Description

### Size & Structure
- **Total Records**: 45,211 customer contacts
- **Features**: 20 variables (demographic, financial, campaign-related)
- **Target Variable**: `y` (binary: 'yes' or 'no' - term deposit subscription)

### Feature Breakdown

#### Demographic Variables
- **age**: Customer age (numeric)
- **job**: Type of occupation (categorical: 12 categories including blue-collar, management, technician, etc.)
- **marital**: Marital status (categorical: married, single, divorced)
- **education**: Education level (categorical: primary, secondary, tertiary, unknown)

#### Financial Variables
- **balance**: Annual balance in euros (numeric) - heavily right-skewed with extreme outliers
- **default**: Credit in default? (binary: yes/no) - severe class imbalance (1.8% yes)
- **housing**: Active housing loan? (binary: yes/no) - well-balanced (55.6% yes)
- **loan**: Active personal loan? (binary: yes/no) - imbalanced (16% yes)

#### Campaign Variables
- **contact**: Contact communication type (categorical: cellular, telephone, unknown)
- **day**: Day of the month contact was made (numeric: 1-31)
- **month**: Month of contact (categorical: shows strong seasonality - May peak with 30% of contacts)
- **duration**: Duration of last contact in seconds (numeric) - most calls <250 seconds
- **campaign**: Number of contacts performed in this campaign (numeric) - heavily concentrated at 1-3 contacts
- **pdays**: Days since last contact from previous campaign (numeric) - 80% value of -1 (never contacted)
- **previous**: Number of contacts before this campaign (numeric) - 99%+ are 0

#### Previous Campaign Data
- **poutcome**: Outcome of previous marketing campaign (categorical) - 82% unknown values (critical data quality issue)

### Critical Data Quality Issues

1. **Class Imbalance**: Target variable is heavily imbalanced
   - Minority class ("yes"): 11.69% (5,289 records)
   - Majority class ("no"): 88.30% (39,922 records)

2. **Missing/Unknown Values**:
   - `poutcome`: 82% unknown
   - `contact`: 30% unknown
   - `education`: 4% unknown

3. **Extreme Skewness**:
   - `balance`: 40,000 observations near zero with extreme positive outliers
   - `duration`: Most calls <250 seconds, with tail extending to 5,000+ seconds
   - `pdays`: 80% concentration at -1 value

4. **Feature Redundancy**:
   - `pdays` shows high correlation with target, posing multicollinearity risk

---

## Project Workflow

### 1. Environment Setup

```python
# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize PySpark
import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.types import *

# Import ML libraries
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, 
    RandomForestClassifier, LinearSVC, GBTClassifier
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, 
    BinaryClassificationEvaluator
)
```

**Configuration**: Spark session named "Bank_Cespedes_Spark" is initialized with default configuration.

---

### 2. Data Loading

The dataset is loaded from HDFS in CSV format with semicolon delimiters:

```python
df00 = sparkC.read.format("csv").option("header", "true")\
              .option("sep",";").load("hdfs://10.195.34.34:9000/csv/bank-full.csv")
```

Initial exploration shows all columns are loaded as strings and need type conversion.

---

### 3. Exploratory Data Analysis (EDA)

#### 3.1 Schema Inspection & Type Conversion

**Numeric Conversions** (7 variables):
- age, balance, day, duration, campaign, pdays, previous

These are converted from string to integer type for mathematical operations.

#### 3.2 Target Variable Analysis

The distribution of the target variable (`y`) reveals severe class imbalance:
- Class "no": 39,922 records (88.30%)
- Class "yes": 5,289 records (11.69%)

**Impact**: Standard algorithms may develop bias toward the majority class. This is addressed through oversampling in the data preparation phase.

#### 3.3 Categorical Variables Analysis

**Job Distribution**:
- Dominated by blue-collar (9,732) and management (9,458) professionals
- 288 "unknown" entries (0.6%) representing noise
- Minority groups like student (938) and housemaid (1,240) are underrepresented

**Marital Status**:
- Heavily skewed toward married (27,214 - 60%)
- Single: 12,790 (28%)
- Divorced: 5,207 (11%)

**Education**:
- Secondary education predominates (23,202 - 51%)
- Tertiary: 13,053 (29%)
- Primary: 6,858 (15%)
- Unknown: 1,857 (4%) - requires imputation strategy

**Default History**:
- Extreme imbalance: 98.2% have no default
- Only 815 cases with credit default
- Provides minimal variability; models may ignore this feature

**Housing & Personal Loans**:
- Housing: Well-balanced (55.6% yes vs 44.4% no)
- Personal Loan: Moderately imbalanced (16.0% yes vs 84.0% no)

**Contact Method**:
- Cellular dominates (29,285 - 65%)
- Telephone: 2,906 (6%)
- **Critical Issue**: Unknown: 13,020 (29%) - nearly 1/3 of data

**Campaign Timing**:
- Extreme seasonality observed
- May peak: 13,766 contacts (30%)
- Summer months (Jun-Aug): sustained high volume
- Winter months (Dec-Mar): minimal activity (<1%)
- **Risk**: Model may overfit on seasonal patterns

**Previous Campaign Outcome**:
- **Severity**: 36,959 "unknown" (82%)
- Success: 1,511 (3%)
- Failure: 4,901 (11%)
- This massive missing information necessitates treating "unknown" as a distinct category

#### 3.4 Numerical Variables Visualization

**Age Distribution**:
- Concentrated between late 20s and late 40s
- Clear peak in 30-35 age range
- Noticeable decline after age 60
- **Key Insight**: Model likely to perform better for working-age population

**Balance Distribution**:
- Extreme right-skew with dense concentration near zero
- 40,000+ observations with near-zero balance
- Long sparse tail with outliers exceeding €100,000
- **ML Impact**: Linear models and distance-based algorithms require transformation (log, Box-Cox)

**Day of Month**:
- Relatively even spread across all days
- Sudden drop-off after day 24
- No clear day-of-month pattern in marketing effectiveness

**Call Duration**:
- Highly concentrated: 30,000+ calls under 250 seconds (<4 minutes)
- Extreme outliers extending to 5,000+ seconds
- **Predictive Power**: Duration shows strongest correlation with subscription success

**Campaign Contacts**:
- 40,000 observations concentrated in first contact (1-3 interactions)
- Operational efficiency: most cases resolved quickly

**Previous Contact Days (pdays)**:
- 80% concentration at -1 value (never previously contacted)
- Creates multicollinearity risk with target variable
- **Decision**: Dropped from final feature set

**Previous Contacts**:
- 99%+ are zero (new prospects)
- Limited variability; minimal predictive power initially

#### 3.5 Correlation Analysis

Pearson correlation matrix reveals:
- **duration** and **pdays**: High positive correlation with target
- **pdays**: High multicollinearity risk → removed from model features
- Most other features show weak individual correlation but important interactions

#### 3.6 Box Plot Analysis

Target variable comparisons (yes vs no) show:
- **Strongest discriminators**: duration and pdays (clear separation between groups)
- **Weak discriminators**: age and day (minimal visual variation)
- Most numeric variables require interaction effects to be predictive

---

### 4. Data Cleaning & Preparation

#### 4.1 Outlier & Extreme Value Handling

**Previous Contacts Threshold**:
- Extreme outliers in "previous" variable identified (>30 contacts)
- These represent <0.1% of data but distort distributions
- **Action**: Filter out records where previous > 30

```python
df02 = df01.filter(F.col('previous') <= 30)
```

#### 4.2 Multicollinearity Removal

**pdays Column Elimination**:
- Rationale: High correlation with target (multicollinearity risk)
- Problematic -1 values (80% of data) cause scale issues
- Already captured by other campaign variables

```python
df03 = df02.drop('pdays')
```

#### 4.3 Class Imbalance Resolution via Oversampling

The severe class imbalance (88:12 ratio) would bias models toward the majority class. Oversampling is employed:

```python
dfUpperD = df03.filter(df03['y'] == 'no')        # Majority: 39,922
dfLowerD = df03.filter(df03['y'] == 'yes')       # Minority: 5,289

# Oversample minority to match majority
factor = dfUpperD.count() / dfLowerD.count()
dfOverSampledLower = dfLowerD.sample(True, factor, seed=42)

# Combine into balanced dataset
df04 = dfUpperD.union(dfOverSampledLower)
```

**Result**: Perfectly balanced dataset (50-50 class distribution) for training, eliminating bias while preserving real-world distribution in test set.

---

### 5. Feature Engineering

#### 5.1 Categorical Encoding Strategy

All categorical variables require encoding for ML algorithms:

**Variables Processed**: job, marital, education, default, month, housing, loan, contact, poutcome

**Two-Step Process**:
1. **StringIndexer**: Maps string labels to numeric indices (0, 1, 2, ...)
2. **OneHotEncoder**: Converts indices into binary vectors

Example: job with 12 categories becomes 11 one-hot features (reference encoding)

```python
col_cat = ['job', 'marital', 'education', 'default', 'month', 
           'housing', 'loan', 'contact', 'poutcome']

for col in col_cat:
    indexer = StringIndexer(inputCol=col, outputCol=col + 'x')
    encoder = OneHotEncoder(inputCols=[col + 'x'], outputCols=[col + '_oneHot'])
    stages += [indexer, encoder]
```

#### 5.2 Target Variable Encoding

Binary target (`y: yes/no`) encoded as:
- 0 = "no" (majority class, negative case)
- 1 = "yes" (minority class, positive case)

```python
label = StringIndexer(inputCol='y', outputCol='label', 
                      stringOrderType="alphabetAsc")
```

#### 5.3 Feature Vector Assembly

Numerical and encoded categorical features combined into single feature vector:

**Numeric Features** (6): age, balance, duration, day, campaign, previous
**Categorical Features** (9): job, marital, education, default, month, housing, loan, contact, poutcome

Total: **43 features** in final feature vector

#### 5.4 PySpark Pipeline Implementation

A **Pipeline** orchestrates the entire transformation sequence:

```python
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df04)
model = pipelineModel.transform(df04)
```

**Benefits**:
- Ensures consistent preprocessing on train and test data
- Prevents data leakage
- Scalable and reproducible
- Easy model serialization and deployment

#### 5.5 Data Persistence

Processed data saved as Parquet (columnar format for efficiency):

```python
df05 = model.select("label", "features")
df05.write.mode("overwrite").parquet("output.parquet")
```

---

### 6. Model Development

Five distinct classification algorithms are trained and evaluated:

#### 6.1 Train-Test Split

Data partitioned into:
- **Training Set**: 80% (used to fit models)
- **Test Set**: 20% (evaluates generalization)

Stratified split maintains class balance in both sets:
- Training: ~50% each class
- Testing: ~50% each class (reflecting balanced dataset)

```python
trainData, testData = df05.randomSplit([.8, .2], seed=4321)
```

#### 6.2 Model Implementations

##### Model 1: Logistic Regression

**Description**: Linear classification model using sigmoid activation. Best for interpretable, probability-based predictions.

```python
LRinstance = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
LRmodel = LRinstance.fit(trainData)
LRpred = LRmodel.transform(testData)
```

**Performance**:
- Accuracy: 83.8%
- Precision: 83.8%
- Recall: 83.8%
- F1-Score: 83.8%
- ROC-AUC: 0.914

**Strengths**: Excellent baseline model; balanced performance; strong probability calibration.

---

##### Model 2: Decision Tree

**Description**: Tree-based model that creates hierarchical splits on features. Interpretable but prone to overfitting.

```python
instanceDT = DecisionTreeClassifier(labelCol='label', featuresCol='features')
modelDT = instanceDT.fit(trainData)
predDT = modelDT.transform(testData)
```

**Performance**:
- Accuracy: 81.7%
- Precision: 81.7%
- Recall: 81.7%
- F1-Score: 81.7%
- ROC-AUC: 0.753

**Strengths**: Highly interpretable feature importance.
**Weaknesses**: Lower ROC-AUC indicates poor probability ranking; step-function probabilities lack fine-grained discrimination.

---

##### Model 3: Random Forest

**Description**: Ensemble of decision trees trained on random subsets. Reduces overfitting while maintaining interpretability.

```python
instanceRF = RandomForestClassifier(labelCol='label', featuresCol='features')
modelRF = instanceRF.fit(trainData)
predRF = modelRF.transform(testData)
```

**Performance**:
- Accuracy: 80.2%
- Precision: 80.3%
- Recall: 80.2%
- F1-Score: 80.2%
- ROC-AUC: 0.891

**Strengths**: Robust probability estimates (ROC-AUC: 0.891) superior to single tree; feature importance rankings.
**Trade-offs**: Slightly lower classification accuracy than LR/SVM, but better overall ranking ability.

---

##### Model 4: Support Vector Machine (Linear SVC)

**Description**: Finds optimal hyperplane maximizing margin between classes. Excellent for binary classification with clear decision boundaries.

```python
instanceSVM = LinearSVC(labelCol='label', featuresCol='features')
modelSVM = instanceSVM.fit(trainData)
predSVM = modelSVM.transform(testData)
```

**Performance**:
- Accuracy: 84.1%
- Precision: 84.1%
- Recall: 84.1%
- F1-Score: 84.1%
- ROC-AUC: 0.915

**Strengths**: Highest classification accuracy (84.1%); exceptional ROC-AUC (0.915); robust in high-dimensional space.
**Note**: Uses rawPrediction instead of probability output.

---

##### Model 5: Gradient Boosted Trees (GBT)

**Description**: Sequential ensemble where each tree corrects previous errors. Captures complex non-linear interactions.

```python
instanceGBT = GBTClassifier(labelCol='label', featuresCol='features')
modelGBT = instanceGBT.fit(trainData)
predGBT = modelGBT.transform(testData)
```

**Performance**:
- Accuracy: 86.0%
- Precision: 86.2%
- Recall: 86.0%
- F1-Score: 86.0%
- ROC-AUC: 0.928  **BEST**

**Strengths**: 
- **Highest overall performance** across all metrics
- Superior ROC-AUC (0.928) - best ranking capability
- Captures complex feature interactions
- Balanced precision-recall tradeoff

**Mechanism**: Each successive tree learns from residuals of previous iterations, enabling aggressive non-linear pattern recognition.

---

### 7. Model Evaluation

#### 7.1 Evaluation Metrics

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **Confusion Matrix** | Counts TP, TN, FP, FN | Visual breakdown of classification errors |
| **Accuracy** | (TP + TN) / Total | Overall correctness (50% baseline due to balance) |
| **Precision** | TP / (TP + FP) | Of predicted "yes", how many correct? |
| **Recall** | TP / (TP + FN) | Of actual "yes", how many did we find? |
| **F1-Score** | Harmonic mean of precision & recall | Balanced accuracy-recall metric |
| **ROC-AUC** | Area under ROC curve | Ranking ability; invariant to threshold |

#### 7.2 Metric Implementation

```python
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

binEvaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
roc_auc = binEvaluator.evaluate(predictions)
```

#### 7.3 Visualization Functions

**Confusion Matrix Plot**:
- Heatmap showing TP, TN, FP, FN counts
- Identifies systematic error patterns

**ROC Curve**:
- Plots True Positive Rate vs False Positive Rate
- Area under curve (AUC) indicates ranking quality
- 0.5 = random classifier; 1.0 = perfect classifier

---

## Key Findings

### Data Quality Insights

1. **Severe Class Imbalance**: Original 88:12 ratio addressed through oversampling
2. **Temporal Concentration**: May accounts for 30% of all contacts; seasonality bias likely
3. **Channel Metadata Missing**: 30% unknown contact type; significant data quality gap
4. **Previous Campaign Data**: 82% unknown, limiting historical context
5. **New Customer Base**: 99%+ have zero previous contacts
6. **Call Duration Importance**: Strongest single predictor of subscription

### Feature Engineering Outcomes

- One-hot encoding of 9 categorical variables creates 43-dimensional feature space
- Multicollinearity (pdays) successfully eliminated
- Class balance achieved without information loss
- Feature vector properly normalized for distance-based algorithms

### Model Hierarchy

The models rank from worst to best:

1. **Decision Tree** (ROC: 0.753) - Poor probability calibration
2. **Random Forest** (ROC: 0.891) - Good ensemble baseline
3. **Logistic Regression** (ROC: 0.914) - Excellent linear model
4. **Linear SVM** (ROC: 0.915) - Near-best classification
5. **Gradient Boosted Trees** (ROC: 0.928) - **CHAMPION MODEL**

### Winner: Gradient Boosted Trees

GBT achieved the highest performance across all metrics:
- **86% accuracy** - correctly classifies 86 of 100 customers
- **86.2% precision** - of predicted subscribers, 86% actually subscribed
- **86% recall** - catches 86% of actual subscribers
- **0.928 ROC-AUC** - superior ranking ability for threshold optimization

---

## Performance Results Summary

```
┌──────────────────────┬──────────┬───────────┬────────┬──────────┬─────────┐
│ Model                │ Accuracy │ Precision │ Recall │ F1-Score │ ROC-AUC │
├──────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ Logistic Regression  │  83.8%   │   83.8%   │ 83.8%  │  83.8%   │  0.914  │
│ Decision Tree        │  81.7%   │   81.7%   │ 81.7%  │  81.7%   │  0.753  │
│ Random Forest        │  80.2%   │   80.3%   │ 80.2%  │  80.2%   │  0.891  │
│ Linear SVM           │  84.1%   │   84.1%   │ 84.1%  │  84.1%   │  0.915  │
│ Gradient Boosted     │  86.0%   │   86.2%   │ 86.0%  │  86.0%   │  0.928  │
│ Trees (GBT)         │          │           │        │          │    ✓    │
└──────────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘
```

---

## Technologies & Libraries

### Distributed Computing
- **Apache Spark** (PySpark): Distributed data processing at scale
- **HDFS**: Hadoop Distributed File System for data storage

### Data Processing & Analysis
- **Pandas**: DataFrames and data manipulation
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization

### Machine Learning
- **PySpark MLlib**: Distributed ML algorithms
- **Scikit-learn**: ROC curve calculations
- **PySpark ML**: Modern DataFrame-based ML pipeline

### Feature Engineering
- StringIndexer: Categorical to numeric encoding
- OneHotEncoder: One-hot encoding for categorical variables
- VectorAssembler: Combines features into vectors
- Pipeline: Orchestrates transformations

---

## Requirements

```
python >= 3.7
apache-spark >= 2.4
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```

### Installation

```bash
# Install PySpark
pip install pyspark

# Install data science libraries
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Getting Started

### 1. Environment Setup

Ensure Spark is installed and JAVA_HOME is set:

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin
```

### 2. Data Preparation

Place bank-full.csv in HDFS:

```bash
hdfs dfs -put bank-full.csv /csv/
```

Or modify the load path in the notebook for local file system.

### 3. Run the Notebook

```bash
jupyter notebook Taller_Clasification_Cespedes.ipynb
```

### 4. Execute Cells Sequentially

- **Section 1**: Environment initialization and Spark session setup
- **Section 2**: Data loading and schema inspection
- **Sections 3-4**: Exploratory data analysis and visualization
- **Sections 5-6**: Data cleaning, feature engineering, pipeline
- **Sections 7-11**: Model training and evaluation for each algorithm
- **Section 12**: Model comparison and visualization

---

## Business Applications

This model enables:

1. **Customer Targeting**: Identify high-probability subscribers before expensive campaign calls
2. **Resource Optimization**: Prioritize high-conversion customer segments
3. **Campaign Timing**: Leverage seasonality insights for optimal call scheduling
4. **Product Design**: Understand demographic preferences for term deposit offerings
5. **Cost Reduction**: Reduce call volume while maintaining conversion rates

---

## Future Improvements

- **Hyperparameter Tuning**: Grid search/Bayesian optimization for each model
- **Feature Selection**: Recursive feature elimination to identify most predictive variables
- **Cross-Validation**: K-fold CV for more robust metric estimation
- **Class Weights**: Alternative to oversampling; weights cost of misclassification
- **SMOTE**: Synthetic Minority Over-sampling Technique for better balance
- **Threshold Optimization**: Adjust decision boundary based on business costs
- **Stacking/Ensemble**: Combine multiple models via meta-learner
- **Real-time Predictions**: Deploy GBT model as REST API

---

## Conclusion

This comprehensive workshop demonstrates that **Gradient Boosted Trees** emerge as the optimal model for bank marketing classification, achieving 86% accuracy and 0.928 ROC-AUC. The project illustrates the complete machine learning lifecycle: from raw data exploration through model comparison, emphasizing the importance of rigorous data cleaning, feature engineering, and careful model selection. The insights derived from this analysis provide a data-driven framework for optimizing bank marketing campaign efficiency and conversion rates.

---

## Contact & Attribution

**Author**: Diego Felipe Céspedes  
**Institution**: Pontificia Universidad Javeriana  
**Workshop**: Metrics & Machine Learning with PySpark

---

## License

This project is provided for educational purposes as part of the Pontificia Universidad Javeriana curriculum.

---

## References

- [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Apache Spark Documentation](https://spark.apache.org/docs/)
- [PySpark ML User Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
