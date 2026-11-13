#!/usr/bin/env python3
"""
descarregar_argoverse.py — imagens das câmaras TRASEIRAS do Argoverse 2 para testar.

O Argoverse 2 (Sensor) está num bucket PÚBLICO da AWS: não precisa de conta nem de
ferramentas extra (só Python). Cada gravação (~15 s, 20 fps) tem, por câmara,
~319 imagens JPG (~90 MB). Câmaras traseiras: ring_rear_left e ring_rear_right
(viradas para trás, ligeiramente para os lados).

Uso (a partir da pasta Projeto/):
  python ferramentas/descarregar_argoverse.py                 # 1 gravação, ring_rear_left
  python ferramentas/descarregar_argoverse.py --n 3 --camara ring_rear_right
Depois, no main.py:  FONTE = r"<pasta impressa no fim>"   (pasta de imagens)

Licença do Argoverse 2: CC BY-NC-SA 4.0 (uso académico/não comercial).
Os dados vão para C:\\Users\\<user>\\va_datasets\\argoverse (fora do OneDrive).
"""
import argparse
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://argoverse.s3.amazonaws.com"
BASE = "datasets/av2/sensor/val/"


def _listar(prefixo, delimitador=None, maximo=1000):
    url = f"{BUCKET}/?list-type=2&prefix={prefixo}&max-keys={maximo}"
    if delimitador:
        url += f"&delimiter={delimitador}"
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    if delimitador:
        return re.findall(r"<Prefix>([^<]+)</Prefix>", xml)[1:]   # 1.º é o próprio prefixo
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="nº de gravações")
    ap.add_argument("--camara", default="ring_rear_left",
                    choices=["ring_rear_left", "ring_rear_right"])
    proj = Path(__file__).resolve().parents[1]
    base = proj / "va_datasets" if (proj / "va_datasets").is_dir() else Path.home() / "va_datasets"
    ap.add_argument("--destino", default=str(base / "argoverse"))
    ap.add_argument("--paralelo", type=int, default=16, help="downloads em simultâneo")
    a = ap.parse_args()

    logs = _listar(BASE, "/", maximo=a.n + 1)[:a.n]
    for log in logs:
        log_id = log.rstrip("/").split("/")[-1]
        pasta = Path(a.destino) / log_id / a.camara
        pasta.mkdir(parents=True, exist_ok=True)
        chaves = [k for k in _listar(f"{log}sensors/cameras/{a.camara}/") if k.endswith(".jpg")]
        faltam = [k for k in chaves if not (pasta / os.path.basename(k)).exists()]
        print(f"[argoverse] {log_id}: {len(chaves)} imagens ({len(faltam)} por descarregar) -> {pasta}")

        def baixar(k):
            dst = pasta / os.path.basename(k)
            tmp = dst.with_suffix(".part")
            for tentativa in range(6):            # o servidor às vezes corta a ligação
                try:
                    urllib.request.urlretrieve(f"{BUCKET}/{k}", tmp)
                    tmp.replace(dst)              # só fica .jpg quando está completo
                    return
                except OSError:
                    time.sleep(1 + 2 * tentativa)
            raise RuntimeError(f"falhou: {k}")

        # em PARALELO: o tempo é dominado pela latência de cada pedido, não pela rede
        with ThreadPoolExecutor(max_workers=a.paralelo) as ex:
            for i, _ in enumerate(ex.map(baixar, faltam), 1):
                if i % 50 == 0 or i == len(faltam):
                    print(f"[argoverse]   {i}/{len(faltam)}", flush=True)
        print(f"\nPronto. No main.py:\n  FONTE = r\"{pasta}\"\n")


if __name__ == "__main__":
    main()
