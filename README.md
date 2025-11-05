# 🧠 Amazon ML Challenge 2025 — Smart Product Pricing Challenge

This repository contains my solution for the **Amazon ML Challenge 2025**, titled **“Smart Product Pricing Challenge”**.  
The goal is to build a **Machine Learning model** that predicts optimal product prices based on textual product information, analyzing patterns in product descriptions, brand, specifications, and quantity.

---

## 🧩 Problem Statement

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction.  
The challenge involves developing an ML model that analyzes product details — including product description, title, and item pack quantity — to predict the product's price.

### **Dataset Details**
| File | Description |
|------|--------------|
| `dataset/train.csv` | Training data with 75,000 products and their prices. |
| `dataset/test.csv` | Test data with 75,000 products (no price). |
| `dataset/sample_test_out.csv` | Sample output file format. |

**Columns:**
- `sample_id`: Unique identifier for each sample  
- `catalog_content`: Text combining title, product description, and item pack quantity  
- `image_link`: Public URL to the product image  
- `price`: Target variable (only in train set)

---

## 🧮 Objective

Develop a robust ML pipeline that:
1. Understands textual product descriptions.
2. Optionally integrates image-based insights.
3. Predicts the product price with minimal **SMAPE (Symmetric Mean Absolute Percentage Error)**.

**Evaluation Metric:**
\[
SMAPE = \frac{1}{n} \sum \frac{|predicted - actual|}{(|predicted| + |actual|)/2}
\]
Lower SMAPE → better model performance.

---

## ⚙️ Approach

### **1. Data Preprocessing**
- Cleaned and normalized product text data.
- Removed unwanted characters, punctuation, and redundant words.
- Tokenized and transformed text using **TF-IDF Vectorization**.

### **2. Feature Engineering**
- Extracted textual patterns and n-grams.
- Generated statistical features such as word counts and IPQ presence.
- (Optional extension): Visual embeddings from product images.

### **3. Model Training**
- Trained three models independently:
  - **LightGBM**
  - **XGBoost**
  - **Ridge Regression**

### **4. Model Ensembling**
- Weighted ensemble based on inverse SMAPE performance.
- Combined predictions for improved generalization.

### **5. Output Generation**
- Created a submission file `test_out.csv` matching the exact format:
