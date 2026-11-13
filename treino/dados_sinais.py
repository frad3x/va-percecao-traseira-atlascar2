"""
dados_sinais.py — descarrega e converte os datasets de sinais para o Ultralytics.

  GTSDB (German Traffic Sign DETECTION Benchmark): 900 imagens 1360x800 com as
        caixas dos sinais -> dataset de DETEÇÃO YOLO (4 categorias ou 43 tipos).
        Divisão oficial: imagens 0-599 treino, 600-899 validação.
  GTSRB (German Traffic Sign RECOGNITION Benchmark): ~39 mil recortes em 43
        classes -> dataset de CLASSIFICAÇÃO YOLO (pasta por classe).
        Dividido por FAIXA (track): cada sinal real aparece em ~30 frames quase
        iguais; separar por faixa evita "cópias" do treino na validação.

Fontes oficiais (servidor ERDA, Univ. de Copenhaga, que aloja os benchmarks):
  GTSDB  ~1,66 GB   GTSRB (treino) ~276 MB

Tudo vai para DATASETS (por omissão C:\\Users\\<user>\\va_datasets), FORA do
OneDrive, para os GB de imagens não serem sincronizados para a nuvem.
Pode mudar-se com a variável de ambiente VA_DATASETS.
"""
import os
import random
import sys
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # para importar percecao/
from percecao.sinais_treinados import CATEGORIA, CATEGORIAS_PT, NOMES_PT  # noqa: E402

# Onde ficam os datasets: VA_DATASETS (se definida) > Projeto/va_datasets (se existir)
# > C:\Users\<user>\va_datasets (fora do OneDrive, o recomendado)
_PROJ = Path(__file__).resolve().parents[1]
_PADRAO = _PROJ / "va_datasets" if (_PROJ / "va_datasets").is_dir() else Path.home() / "va_datasets"
DATASETS = Path(os.environ.get("VA_DATASETS", str(_PADRAO)))

URL_GTSDB = ("https://sid.erda.dk/public/archives/"
             "ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip")
URL_GTSRB = ("https://sid.erda.dk/public/archives/"
             "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip")


# ----------------------------- utilitários -----------------------------
def _tamanho_remoto(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def descarregar(url, destino, tentativas=40):
    """Download ROBUSTO: confirma o tamanho final e, se o servidor cortar a
    ligação a meio (acontece com ficheiros grandes), RETOMA a partir do ponto
    onde parou (HTTP Range) em vez de recomeçar."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    total = _tamanho_remoto(url)
    tmp = destino.with_suffix(destino.suffix + ".part")
    if destino.exists():
        if (not total or destino.stat().st_size == total) and zipfile.is_zipfile(destino):
            print(f"[dados] já existe (completo): {destino.name}")
            return destino
        destino.replace(tmp)          # incompleto de uma execução anterior -> retomar

    print(f"[dados] a descarregar {url} ({total / 2**20:.0f} MB)")
    prox = 0.1
    for tent in range(1, tentativas + 1):
        feito = tmp.stat().st_size if tmp.exists() else 0
        if total and feito >= total:
            break
        cab = {"Range": f"bytes={feito}-"} if feito else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=cab),
                                        timeout=60) as r:
                retoma = feito and r.status == 206
                if feito and not retoma:
                    feito = 0                       # servidor ignorou o Range -> recomeçar
                with open(tmp, "ab" if retoma else "wb") as f:
                    while True:
                        bloco = r.read(1 << 20)
                        if not bloco:
                            break
                        f.write(bloco)
                        feito += len(bloco)
                        if total and feito / total >= prox:
                            print(f"[dados]   {100 * feito / total:3.0f}%  "
                                  f"({feito / 2**20:.0f} MB)", flush=True)
                            prox = (int(10 * feito / total) + 1) / 10
        except Exception as e:  # noqa: BLE001  (ligação caída, timeout, reset...)
            print(f"[dados]   ligação interrompida ({type(e).__name__}); "
                  f"a retomar aos {feito / 2**20:.0f} MB ({tent}/{tentativas})", flush=True)
            time.sleep(min(3 * tent, 30))
            continue
        if total and tmp.stat().st_size < total:    # fim "normal" mas incompleto
            print(f"[dados]   o servidor cortou aos {tmp.stat().st_size / 2**20:.0f} MB; "
                  f"a retomar ({tent}/{tentativas})", flush=True)
            continue
        break

    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"Download incompleto ({tmp.stat().st_size} de {total} bytes). "
                           "Volta a correr: retoma de onde parou.")
    tmp.replace(destino)
    if not zipfile.is_zipfile(destino):
        raise RuntimeError(f"{destino.name} não é um zip válido.")
    print(f"[dados] download completo: {destino.name}")
    return destino


def extrair(zip_path, pasta):
    marca = Path(pasta) / ".extraido"
    if marca.exists():
        return
    print(f"[dados] a extrair {Path(zip_path).name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pasta)
    marca.touch()


def slug(txt):
    """Nome só com ASCII (para pastas): 'cedência de passagem' -> 'cedencia_de_passagem'."""
    s = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def _escrever_yaml(caminho, raiz, nomes):
    linhas = [f"path: {Path(raiz).as_posix()}", "train: images/train", "val: images/val", "names:"]
    linhas += [f"  {i}: {slug(n)}" for i, n in enumerate(nomes)]
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


# ----------------------------- GTSDB (deteção) -----------------------------
def preparar_gtsdb(modo="categorias"):
    """Devolve o caminho do dados.yaml do dataset de deteção."""
    assert modo in ("categorias", "todas")
    out = DATASETS / f"gtsdb_yolo_{modo}"
    yaml = out / "dados.yaml"
    if yaml.exists():                 # já convertido: os ficheiros brutos já não são precisos
        print(f"[dados] GTSDB já convertido: {yaml}")
        return yaml
    base = DATASETS / "gtsdb"
    zipf = descarregar(URL_GTSDB, base / "FullIJCNN2013.zip")
    extrair(zipf, base)
    src = base / "FullIJCNN2013"

    anot = {}
    for linha in (src / "gt.txt").read_text().splitlines():
        if not linha.strip():
            continue
        nome, x1, y1, x2, y2, c = linha.strip().split(";")
        anot.setdefault(nome, []).append((int(x1), int(y1), int(x2), int(y2), int(c)))

    n_sinais = 0
    for i in range(900):
        nome = f"{i:05d}.ppm"
        img = cv2.imread(str(src / nome))
        if img is None:
            continue
        split = "train" if i < 600 else "val"
        h, w = img.shape[:2]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{i:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        linhas = []
        for x1, y1, x2, y2, c in anot.get(nome, []):   # imagens sem sinais ficam vazias
            k = CATEGORIA[c] if modo == "categorias" else c
            linhas.append(f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                          f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        n_sinais += len(linhas)
        (out / "labels" / split / f"{i:05d}.txt").write_text("\n".join(linhas))

    _escrever_yaml(yaml, out, CATEGORIAS_PT if modo == "categorias" else NOMES_PT)
    print(f"[dados] GTSDB convertido: 900 imagens, {n_sinais} sinais -> {yaml}")
    return yaml


# ----------------------------- GTSRB (classificação) -----------------------------
def preparar_gtsrb(frac_val=0.15, semente=0):
    """Devolve a pasta do dataset de classificação (train/ e val/ com uma pasta por classe)."""
    out = DATASETS / "gtsrb_cls"
    if (out / ".pronto").exists():    # já convertido: os ficheiros brutos já não são precisos
        print(f"[dados] GTSRB já convertido: {out}")
        return out
    base = DATASETS / "gtsrb"
    zipf = descarregar(URL_GTSRB, base / "GTSRB_Final_Training_Images.zip")
    extrair(zipf, base)
    src = base / "GTSRB" / "Final_Training" / "Images"

    rng = random.Random(semente)
    n_tr = n_va = 0
    for c in range(43):
        ficheiros = sorted((src / f"{c:05d}").glob("*.ppm"))
        faixas = sorted({f.name.split("_")[0] for f in ficheiros})
        rng.shuffle(faixas)
        val = set(faixas[:max(1, round(len(faixas) * frac_val))])
        pasta = f"{c:02d}_{slug(NOMES_PT[c])}"      # prefixo 00..42 -> ordem = ClassId
        for f in ficheiros:
            split = "val" if f.name.split("_")[0] in val else "train"
            dst = out / split / pasta / (f.stem + ".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(f))
            if img is not None:
                cv2.imwrite(str(dst), img)
                if split == "val":
                    n_va += 1
                else:
                    n_tr += 1
        print(f"[dados]   classe {c:02d} ({NOMES_PT[c]}) ok", flush=True)
    (out / ".pronto").touch()
    print(f"[dados] GTSRB convertido: {n_tr} treino / {n_va} validação -> {out}")
    return out
