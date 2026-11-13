#!/usr/bin/env python3
"""
treinar_sinais.py — treina o reconhecimento de SINAIS (basta dar Run).

Faz tudo, por esta ordem (cada passo é saltado se já estiver feito):
  1) descarrega e converte os datasets (GTSDB + GTSRB) para C:\\Users\\<user>\\va_datasets
  2) treina o DETETOR  (YOLO no GTSDB)  -> encontra sinais + categoria
  3) treina o CLASSIFICADOR (YOLO-cls no GTSRB, 43 tipos) -> diz qual é o sinal
  4) copia os melhores pesos para modelos/sinais_det.pt e modelos/sinais_cls.pt

A partir daí o main.py usa-os automaticamente (e mostra "limite 50", "cedência",
"STOP", "rotunda"... em vez de só "stop sign").

Tempo típico com GPU (RTX 5060): ~1 h no detetor + ~5 min no classificador.
Opções:  --so-preparar   só descarrega/converte os dados
         --rapido        teste rápido (1 época, poucos dados) — NÃO substitui os modelos
"""
import argparse
import os
import shutil
from pathlib import Path

# ========================= CONFIG (edita e dá Run) =========================
MODO = "categorias"      # detetor: "categorias" (4: proibição/perigo/obrigação/outros,
                         #   mais robusto com poucos dados) ou "todas" (43 tipos direto)
BASE_DET = "yolo11s.pt"  # modelo de partida do detetor (transfer learning)
EPOCAS_DET = 80          # épocas máximas (pára antes se deixar de melhorar)
PACIENCIA = 20           # early stopping: épocas sem melhoria antes de parar
IMGSZ_DET = 1280         # resolução de treino (sinais são pequenos -> alta resolução)
LOTE_DET = -1            # tamanho do lote (-1 = automático conforme a memória da GPU)

BASE_CLS = "yolo11n-cls.pt"  # classificador (rápido; os recortes são pequenos)
EPOCAS_CLS = 15
IMGSZ_CLS = 64
LOTE_CLS = 256
# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-preparar", action="store_true", help="só prepara os dados")
    ap.add_argument("--rapido", action="store_true", help="teste rápido (não substitui modelos)")
    a = ap.parse_args()

    raiz = Path(__file__).resolve().parent
    os.chdir(raiz)
    from treino import dados_sinais as ds

    # --- 1) dados ---
    yaml_det = ds.preparar_gtsdb(MODO)
    pasta_cls = ds.preparar_gtsrb()
    if a.so_preparar:
        print("\nDados prontos. Corre sem --so-preparar para treinar.")
        return

    from ultralytics import YOLO
    runs = ds.DATASETS / "runs"           # artefactos de treino FORA do OneDrive
    modelos = raiz / "modelos"
    modelos.mkdir(exist_ok=True)
    rap = a.rapido
    sufixo = "_teste" if rap else ""

    # --- 2) detetor ---
    print("\n========== TREINO DO DETETOR (GTSDB) ==========")
    det = YOLO(BASE_DET)
    det.train(data=str(yaml_det),
              epochs=1 if rap else EPOCAS_DET,
              patience=PACIENCIA,
              imgsz=640 if rap else IMGSZ_DET,
              batch=8 if rap else LOTE_DET,
              fraction=0.2 if rap else 1.0,
              # espelhar só é seguro com categorias (com 43 tipos troca esq./dir.)
              fliplr=0.5 if MODO == "categorias" else 0.0,
              project=str(runs), name="sinais_det" + sufixo, exist_ok=True)
    best_det = Path(det.trainer.best)
    m = det.trainer.metrics or {}
    print(f"Detetor: mAP50 = {m.get('metrics/mAP50(B)', float('nan')):.3f} | "
          f"mAP50-95 = {m.get('metrics/mAP50-95(B)', float('nan')):.3f}")

    # --- 3) classificador ---
    print("\n========== TREINO DO CLASSIFICADOR (GTSRB) ==========")
    cls = YOLO(BASE_CLS)
    cls.train(data=str(pasta_cls),
              epochs=1 if rap else EPOCAS_CLS,
              imgsz=IMGSZ_CLS, batch=LOTE_CLS,
              fraction=0.2 if rap else 1.0,
              fliplr=0.0,              # NUNCA espelhar: "virar à esq." viraria "à dir."
              project=str(runs), name="sinais_cls" + sufixo, exist_ok=True)
    best_cls = Path(cls.trainer.best)
    mc = cls.trainer.metrics or {}
    print(f"Classificador: exatidão top-1 = {mc.get('metrics/accuracy_top1', float('nan')):.3f}")

    # --- 4) instalar os modelos no projeto ---
    if rap:
        print(f"\n[teste rápido] modelos NÃO instalados. Pesos em:\n  {best_det}\n  {best_cls}")
        return
    shutil.copy2(best_det, modelos / "sinais_det.pt")
    shutil.copy2(best_cls, modelos / "sinais_cls.pt")
    print(f"\nModelos instalados em {modelos}:\n  sinais_det.pt\n  sinais_cls.pt")
    print("O main.py passa a usá-los automaticamente.")


if __name__ == "__main__":
    main()
