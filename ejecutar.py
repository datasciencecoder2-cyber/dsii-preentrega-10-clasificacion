"""Alternativa CLI al notebook: mismo entrenamiento, métricas y matriz."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from src.modelado import cargar_datos, construir_pipeline, evaluar

BASE = Path(__file__).resolve().parent


def ejecutar() -> dict:
    X, y, raw = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    metr = evaluar(y_test, pred)
    metadatos = {"origen": "simulado", "filas_totales": len(raw),
                 "filas_train": len(X_train), "filas_test": len(X_test),
                 "prevalencia_churn_test": round(float(y_test.mean()), 4),
                 **metr}
    destino = BASE / "evidencia"
    destino.mkdir(exist_ok=True)
    (destino / "metricas.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (destino / "reporte_clasificacion.txt").write_text(
        classification_report(y_test, pred, labels=[0, 1], target_names=["Permanece", "Abandona"], zero_division=0), encoding="utf-8"
    )
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0,1], display_labels=["Permanece", "Abandona"], cmap="Blues", values_format="d")
    plt.title("Churn | Matriz de confusión (test 20%)")
    plt.tight_layout()
    plt.savefig(destino / "matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Dataset simulado: {len(raw)} | Train={len(X_train)} | Test={len(X_test)}")
    for x in ["accuracy", "precision", "recall", "f1"]:
        print(f"{x.upper():10s}: {metr[x]:.4f}")
    print(f"Matriz TN={metr['tn']} FP={metr['fp']} FN={metr['fn']} TP={metr['tp']}")
    print("EVALUACION COMPLETADA: OK")
    return metr


if __name__ == "__main__":
    ejecutar()
