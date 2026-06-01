
# =========================================
# TELCO CUSTOMER CHURN ANALYSIS PROJECT
# PYTHON PROGRAMLAMA DÖNEM ÖDEVİ
# =========================================

# =========================
# DOSYA YÜKLEME
# =========================
from google.colab import files

uploaded = files.upload()

# =========================
# KÜTÜPHANELER
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# VERİ SETİNİ OKUMA
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =========================
# İLK 5 SATIR
# =========================
print("\n=== FIRST 5 ROWS ===\n")

print(df.head())

# =========================
# VERİ BİLGİLERİ
# =========================
print("\n=== DATA INFO ===\n")

print(df.info())

# =========================
# SAYISAL ÖZET
# =========================
print("\n=== DESCRIBE ===\n")

print(df.describe())

# =========================
# CHURN DAĞILIMI
# =========================
print("\n=== CHURN DISTRIBUTION ===\n")

print(df["Churn"].value_counts())

# =========================
# GRAFİK
# =========================
df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")

plt.xlabel("Churn")

plt.ylabel("Customer Count")

plt.show()

# =========================
# GEREKSİZ SÜTUNU SİL
# =========================
df.drop("customerID", axis=1, inplace=True)

# =========================
# TotalCharges SAYISALA ÇEVİR
# =========================
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# =========================
# EKSİK VERİ KONTROLÜ
# =========================
print("\n=== MISSING VALUES ===\n")

print(df.isnull().sum())

# =========================
# EKSİK VERİLERİ TEMİZLE
# =========================
df.dropna(inplace=True)

# =========================
# KATEGORİK VERİLERİ SAYISALA ÇEVİR
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# X VE y AYIRMA
# =========================
X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]

# =========================
# TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# LOGISTIC REGRESSION
# =========================================
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# =========================
# TAHMİN
# =========================
y_pred = model.predict(X_test)

# =========================
# SONUÇLAR
# =========================
print("\n===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================\n")

print("Accuracy:")

print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nAUC:")

print(roc_auc_score(y_test, y_pred))

# =========================================
# RANDOM FOREST
# =========================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# =========================
# RANDOM FOREST SONUÇLARI
# =========================
print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================\n")

print("Accuracy:")

print(accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, rf_pred))

print("\nAUC:")

print(roc_auc_score(y_test, rf_pred))