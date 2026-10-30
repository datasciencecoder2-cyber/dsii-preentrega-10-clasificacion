"""Descarga OPTATIVA del dataset original publicado por IBM.

El proyecto se ejecuta sin red sobre datos ficticios incluidos en el ZIP.
Nunca sustituye ni altera automáticamente esos datos.
"""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUT = Path(__file__).resolve().parent / "datos" / "Telco-Customer-Churn-IBM.csv"

if __name__ == "__main__":
    print("Descargando muestra pública de IBM (requiere Internet)...")
    try:
        with urlopen(URL, timeout=30) as response:
            raw = response.read()
        digest = sha256(raw).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"SHA-256 inesperado: {digest}. No se guardó el archivo.")
        OUT.write_bytes(raw)
        print(f"Descarga validada: {OUT}")
    except Exception as exc:
        print(f"No se pudo descargar: {exc}")
        print("Podés continuar con el dataset simulado que acompaña la entrega.")
        raise SystemExit(1)
