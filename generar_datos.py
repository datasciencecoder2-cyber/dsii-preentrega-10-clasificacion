"""Recrear un dataset SIMULADO con columnas compatibles con IBM Telco Churn."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def generar(n: int = 2400, semilla: int = 20261030) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[.56, .23, .21])
    tenure = np.empty(n, dtype=int)
    for i, c in enumerate(contract):
        if c == "Month-to-month":
            tenure[i] = rng.integers(0, 65)
        elif c == "One year":
            tenure[i] = rng.integers(6, 73)
        else:
            tenure[i] = rng.integers(15, 73)
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[.35, .49, .16])
    phone = rng.choice(["Yes", "No"], size=n, p=[.9, .1])
    senior = rng.binomial(1, .16, size=n)
    partner = rng.choice(["Yes", "No"], size=n, p=[.48, .52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[.3, .7])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], size=n, p=[.37, .23, .20, .20]).astype(object)
    gender = rng.choice(["Female", "Male"], size=n)
    paperless = rng.choice(["Yes", "No"], size=n, p=[.58, .42])

    def srv(prob):
        yes = rng.random(n) < prob
        return np.where(internet == "No", "No internet service", np.where(yes, "Yes", "No"))

    online_security = srv(.29)
    online_backup = srv(.40)
    device_protection = srv(.41)
    tech_support = srv(.29)
    streaming_tv = srv(.46)
    streaming_movies = srv(.47)
    multiple = np.where(phone == "No", "No phone service", np.where(rng.random(n) < .5, "Yes", "No"))

    charges = np.clip(
        22 + 28*(internet == "DSL") + 49*(internet == "Fiber optic")
        + 7*(multiple == "Yes") + 6*(streaming_tv == "Yes") + 6*(streaming_movies == "Yes")
        + rng.normal(0, 6, size=n), 18, 125
    ).round(2)
    total = np.maximum(0, tenure * charges + rng.normal(0, 35, size=n)).round(2).astype(object)
    total[tenure == 0] = " "  # mismo caso de borde que el archivo IBM

    # Relación probabilística INTENCIONAL; churn no es una regla determinista.
    logits = (
        -1.6 + 1.0*(contract == "Month-to-month") - .8*(contract == "Two year")
        - .024*(tenure - 25) + .012*(charges - 65) + .39*(payment == "Electronic check")
        + .31*(internet == "Fiber optic") + .28*(tech_support == "No")
        + .20*(online_security == "No") + .22*senior
    )
    probability = 1 / (1 + np.exp(-logits))
    churn = np.where(rng.random(n) < probability, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"SIM-{i:06d}" for i in range(1, n+1)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": charges,
        "TotalCharges": total, "Churn": churn,
    })
    # Nulos numéricos y categóricos como casos reales de preprocesamiento.
    df.loc[rng.choice(n, min(n, 30), replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(n, min(n, 24), replace=False), "PaymentMethod"] = np.nan
    return df


if __name__ == "__main__":
    out = BASE / "datos" / "clientes_churn_simulados.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = generar()
    data.to_csv(out, index=False)
    print(f"Dataset SIMULADO generado: {len(data)} clientes; Churn={sum(data.Churn=='Yes')}")
    print(out)
