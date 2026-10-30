"""Herramientas reproducibles para el experimento de churn.

Datos de ejemplo ficticios. No usar para decisiones comerciales reales.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "datos" / "clientes_churn_simulados.csv"

NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERICAS + CATEGORICAS


def cargar_datos(path: str | Path = DATASET) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Lee el CSV y valida el target; NO ajusta transformaciones."""
    df = pd.read_csv(path, dtype={"customerID": "string"})
    requeridas = {"customerID", "Churn", *FEATURES}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {sorted(faltantes)}")
    if df["customerID"].duplicated().any():
        raise ValueError("customerID duplicado")
    if df["Churn"].isna().any() or not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Valores invalidos en Churn")
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAS:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
    X = df[FEATURES].copy()
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype("int64")
    return X, y, df


def construir_pipeline() -> Pipeline:
    """Todo el preprocesamiento se entrenará solamente tras el split."""
    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocesador = ColumnTransformer([
        ("numerico", numerico, NUMERICAS),
        ("categorico", categorico, CATEGORICAS),
    ])
    return Pipeline([
        ("preprocesador", preprocesador),
        ("clasificador", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def evaluar(y_real, predicciones) -> dict:
    """Devuelve métricas de clasificación con clase positiva churn=1."""
    matriz = confusion_matrix(y_real, predicciones, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    return {
        "accuracy": float(accuracy_score(y_real, predicciones)),
        "precision": float(precision_score(y_real, predicciones, zero_division=0)),
        "recall": float(recall_score(y_real, predicciones, zero_division=0)),
        "f1": float(f1_score(y_real, predicciones, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_prueba": int(len(y_real)),
    }
