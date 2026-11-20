#!/usr/bin/env python3
"""
sinais_pt.py — classificador dos SINAIS PORTUGUESES (Regulamento de Sinalização do
Trânsito: códigos A1a, B2, C13…, D1a…), treinado com imagens SINTÉTICAS.

Basta dar Run (ou python treino/sinais_pt.py). Passos (cada um é saltado se já feito):
  1) descarrega os sinais oficiais de perigo (A), prioridade (B), proibição (C) e
     obrigação (D) do Wikimedia Commons (licenças livres; autor e licença de cada um
     ficam em va_datasets/sinais_pt/creditos.csv) e os nomes da Wikipédia (PT);
  2) gera imagens sintéticas a partir de cada sinal: perspetiva, escala, desfocagem,
     luz, ruído, compressão JPEG e fundos de ruas reais (imagens do GTSDB);
  3) treina um YOLO-cls (nunca espelha: "virar à esquerda" não pode virar "à direita");
  4) AVALIA em fotografias REAIS: os recortes do GTSRB dos tipos que existem em Portugal;
  5) instala modelos/sinais_pt_cls.pt (o pipeline usa-o automaticamente).

Porquê sintético: há UMA imagem limpa por sinal; o modelo só aprende a reconhecê-lo
numa fotografia se o treino lhe mostrar o sinal visto de lado, ao longe, desfocado,
à sombra, etc. A avaliação em fotografias reais (passo 4) mostra se isso resultou.

Opções:  --so-dados   só descarrega e gera   |   --rapido   teste de 1 época
"""
import argparse
import csv
import html
import json
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from percecao.fontes import gravar_imagem, ler_imagem  # noqa: E402

PASTA = RAIZ / "va_datasets" / "sinais_pt"
SINAIS = PASTA / "sinais"              # um PNG (com transparência) por sinal
SINT = PASTA / "sintetico"             # dataset gerado (não vai para o git; regenera-se)
FUNDOS = RAIZ / "va_datasets" / "gtsdb_yolo_categorias" / "images" / "train"
GTSRB_VAL = RAIZ / "va_datasets" / "gtsrb_cls" / "val"

CATEGORIAS = ["SVG warning road signs of Portugal", "SVG priority road signs of Portugal",
              "SVG prohibitory road signs of Portugal", "SVG mandatory road signs of Portugal"]
UA = {"User-Agent": "va-percecao-treino/1.0 (projeto academico; python-urllib)"}

# ========================= CONFIG =========================
N_TREINO, N_VAL = 160, 30      # imagens sintéticas por sinal
TAM = 96                       # px das imagens geradas (o treino usa IMGSZ)
BASE, EPOCAS, IMGSZ, LOTE = "yolo11n-cls.pt", 40, 64, 256
# ==========================================================

# GTSRB (fotografias reais) -> código PT, para a avaliação. Só os tipos com equivalente
# claro (o 24 "estreitamento à direita" fica de fora: não é certo se é o A4b ou o A4c).
from percecao.sinais_treinados import GTSRB_PARA_PT  # noqa: E402

# nomes que a Wikipédia não tem (vêm da descrição no Commons ou da própria imagem)
NOMES_EXTRA = {
    "A17a": "Passagem de velocípedes", "A19c": "Linces-ibéricos", "A19d": "Anfíbios",
    "C3s": "Trânsito proibido a automóveis pesados de passageiros",
    "C4g": "Trânsito proibido a mercadorias e a pesados de passageiros",
    "D5c": "Via obrigatória para motociclos", "D6a": "Via reservada a veículos com alta ocupação",
    "D11c": "Fim da via obrigatória para motociclos",
    "D17": "Fim da via reservada a veículos com alta ocupação",
}
EXCLUIR = {"C20g"}             # imagem igual à do C22: duas classes iguais só confundem
LIMITES = (20, 30, 40, 50, 60, 70, 80, 90, 100, 120)   # C13-<v>: gerados a partir do C13-50


# ----------------------------------------------------------------------------
def _api(base, **kw):
    kw.update(format="json", formatversion=2)
    url = base + "?" + urllib.parse.urlencode(kw)
    for t in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 + 3 * t)
    raise RuntimeError(f"falhou: {url[:120]}")


def _nomes_wikipedia():
    """Código -> nome curto, da galeria da Wikipédia 'Sinalização de trânsito em Portugal'."""
    w = _api("https://pt.wikipedia.org/w/api.php", action="parse",
             page="Sinalização de trânsito em Portugal", prop="wikitext")["parse"]["wikitext"]
    nomes = {}
    for m in re.finditer(r"File:Portugal road sign ([^|\]]+?)\.(?:svg|png)\|\s*([^\n]+)", w):
        cod, txt = m.group(1).strip(), m.group(2)
        txt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", txt)   # [[alvo|texto]] -> texto
        txt = re.sub(r"<[^>]+>|'''|''", "", txt)
        nome = re.split(r"\s*[:;]\s*", re.sub(r"^\S+\s*[-–—]\s*", "", txt), maxsplit=1)[0]
        nomes.setdefault(cod, nome.strip()[:60])
    return nomes


def descarregar():
    """Passo 1: PNG de cada sinal + creditos.csv. Devolve {código: nome}."""
    cred = PASTA / "creditos.csv"
    if cred.exists():
        with open(cred, encoding="utf-8") as f:
            return {r["codigo"]: r["nome"] for r in csv.DictReader(f)}
    SINAIS.mkdir(parents=True, exist_ok=True)
    nomes = _nomes_wikipedia()
    titulos = []
    for cat in CATEGORIAS:
        r = _api("https://commons.wikimedia.org/w/api.php", action="query", list="categorymembers",
                 cmtitle="Category:" + cat, cmtype="file", cmlimit=500)
        titulos += [m["title"] for m in r["query"]["categorymembers"] if m["title"].endswith(".svg")]
    linhas = []
    for i in range(0, len(titulos), 40):
        r = _api("https://commons.wikimedia.org/w/api.php", action="query", prop="imageinfo",
                 titles="|".join(titulos[i:i + 40]), iiprop="url|extmetadata", iiurlwidth=256)
        for p in r["query"]["pages"]:
            ii = p["imageinfo"][0]
            em = ii.get("extmetadata", {})
            cod = re.sub(r"^File:Portugal road sign |\.svg$", "", p["title"]).strip()
            destino = SINAIS / f"{cod}.png"
            if not destino.exists():
                req = urllib.request.Request(ii["thumburl"], headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    destino.write_bytes(resp.read())
                time.sleep(0.2)                         # educado com o servidor
            autor = html.unescape(re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", ""))).strip()
            linhas.append(dict(codigo=cod, nome=nomes.get(cod, cod),
                               licenca=em.get("LicenseShortName", {}).get("value", ""),
                               autor=autor, pagina=ii.get("descriptionurl", "")))
    linhas = [r for r in linhas if r["codigo"] not in EXCLUIR]
    for r in linhas:
        if r["codigo"] in NOMES_EXTRA:
            r["nome"] = NOMES_EXTRA[r["codigo"]]
    base = next(r for r in linhas if r["codigo"] == "C13-50")
    for v in LIMITES:                               # limites de velocidade 20..120
        cod = f"C13-{v}"
        if not (SINAIS / f"{cod}.png").exists():
            _limite(SINAIS / "C13-50.png", v, SINAIS / f"{cod}.png")
        if all(r["codigo"] != cod for r in linhas):
            linhas.append(dict(base, codigo=cod, pagina=base["pagina"] + f" (número trocado para {v})"))
    for r in linhas:
        if r["codigo"].startswith("C13-"):
            r["nome"] = f"Limite de velocidade {r['codigo'][4:]} km/h"
    linhas.sort(key=lambda r: _ordem(r["codigo"]))
    with open(cred, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"[sinais_pt] {len(linhas)} sinais descarregados -> {SINAIS}")
    return {r["codigo"]: r["nome"] for r in linhas}


def _limite(base_png, valor, destino):
    """C13 com outro número: tapa o '50' com o fundo branco e escreve o novo valor."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(base_png).convert("RGBA")
    W, H = im.size
    d = ImageDraw.Draw(im)
    r = 0.360 * W                                   # interior branco (dentro do anel vermelho)
    d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=(255, 255, 255, 255))
    txt = str(valor)
    larg = 0.56 * W if len(txt) == 2 else 0.62 * W
    for nome in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            f = ImageFont.truetype(nome, 10)
            break
        except OSError:
            f = None
    tam = 10
    while f is not None and d.textlength(txt, font=ImageFont.truetype(f.path, tam + 2)) < larg:
        tam += 2
    fonte = ImageFont.truetype(f.path, tam) if f is not None else ImageFont.load_default()
    d.text((W / 2, H / 2 + 0.01 * H), txt, font=fonte, fill=(0, 0, 0, 255), anchor="mm")
    im.save(destino)


def _ordem(cod):
    m = re.match(r"([A-Z]+)(\d+)([a-z]?)-?(\d*)", cod)
    if not m:
        return (cod, 0, "", 0)
    return (m.group(1), int(m.group(2)), m.group(3), int(m.group(4) or 0))


def nome_pasta(i, cod, nome):
    s = re.sub(r"[^a-z0-9]+", "_", nome.lower().encode("ascii", "ignore").decode())[:40].strip("_")
    return f"{i:03d}_{cod}__{s}"


# ------------------------------ geração sintética ------------------------------
def _ler_rgba(caminho):
    img = ler_imagem(caminho, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def _fundo(rng, fundos, tam):
    if fundos:
        im = None
        while im is None:
            im = ler_imagem(rng.choice(fundos))
        h, w = im.shape[:2]
        s = rng.randint(tam, max(tam + 1, min(h, w) // 2))
        y, x = rng.randint(0, h - s), rng.randint(0, w - s)
        return cv2.resize(im[y:y + s, x:x + s], (tam, tam), interpolation=cv2.INTER_AREA)
    return np.full((tam, tam, 3), rng.randint(60, 200), np.uint8)


def sintetica(sinal, rng, fundos, tam=TAM):
    """Uma imagem do sinal como apareceria num recorte do detetor."""
    # 1) perspetiva: o sinal visto de lado/de baixo, rodado, com o tamanho de um recorte
    h, w = sinal.shape[:2]
    lado = rng.uniform(0.62, 0.86) * tam           # recorte com margem (como o detetor)
    esc = lado / max(h, w)
    cx, cy = tam / 2 + rng.uniform(-0.08, 0.08) * tam, tam / 2 + rng.uniform(-0.08, 0.08) * tam
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    ang = np.deg2rad(rng.uniform(-8, 8))
    yaw, pit = rng.uniform(-0.35, 0.35), rng.uniform(-0.15, 0.15)   # encurtamento
    pts = []
    for x, y in src:
        u, v = (x - w / 2) * esc, (y - h / 2) * esc
        u *= 1 - abs(yaw) * 0.6
        v *= 1 - abs(pit) * 0.6
        u, v = u * (1 + yaw * v / lado), v * (1 + pit * u / lado)
        pts.append([cx + u * np.cos(ang) - v * np.sin(ang), cy + u * np.sin(ang) + v * np.cos(ang)])
    M = cv2.getPerspectiveTransform(src, np.float32(pts))
    warp = cv2.warpPerspective(sinal, M, (tam, tam), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # 2) cor do sinal: desbotado, mais escuro/claro, temperatura de cor
    rgb = warp[:, :, :3].astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.55, 1.1)
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    rgb *= np.array([rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.15)])
    if rng.random() < 0.3:                           # sombra/brilho num lado do sinal
        g = np.linspace(rng.uniform(0.5, 1.0), rng.uniform(0.9, 1.3), tam, dtype=np.float32)
        rgb *= (g[None, :, None] if rng.random() < 0.5 else g[:, None, None])
    a = warp[:, :, 3:4].astype(np.float32) / 255.0
    # 3) composição sobre o fundo (rua real) e luz global
    img = _fundo(rng, fundos, tam).astype(np.float32) * (1 - a) + rgb * a
    img = img * rng.uniform(0.45, 1.35) + rng.uniform(-30, 30)
    gama = rng.uniform(0.7, 1.4)
    img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** gama
    if rng.random() < 0.15:                          # oclusão parcial (ramo, poste)
        x0 = rng.randint(0, tam - 10)
        cv2.rectangle(img, (x0, 0), (x0 + rng.randint(4, 14), tam), tuple(float(rng.randint(20, 90))
                      for _ in range(3)), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 4) ótica e sensor: baixa resolução (sinal ao longe), desfocagem, movimento, ruído, JPEG
    if rng.random() < 0.7:
        r = rng.randint(14, 48)
        img = cv2.resize(cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA), (tam, tam),
                         interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.25:
        k = rng.randint(3, 7)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(ker, rot, (k, k)))
    ruido = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, 9), img.shape)
    img = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 92)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gerar(nomes, n_treino=N_TREINO, n_val=N_VAL, semente=0):
    """Passo 2: dataset sintético em SINT/{train,val}/<classe>/."""
    if (SINT / ".pronto").exists():
        print(f"[sinais_pt] dataset sintético já existe: {SINT}")
        return SINT
    if SINT.exists():
        shutil.rmtree(SINT)
    rng = random.Random(semente)
    fundos = sorted(FUNDOS.glob("*.jpg")) if FUNDOS.is_dir() else []
    cods = sorted(nomes, key=_ordem)
    for i, cod in enumerate(cods):
        sinal = _ler_rgba(SINAIS / f"{cod}.png")
        if sinal is None:
            continue
        pasta = nome_pasta(i, cod, nomes[cod])
        for split, n in (("train", n_treino), ("val", n_val)):
            d = SINT / split / pasta
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                gravar_imagem(d / f"{k:04d}.jpg", sintetica(sinal, rng, fundos),
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
        if (i + 1) % 20 == 0:
            print(f"[sinais_pt]   {i + 1}/{len(cods)} sinais gerados", flush=True)
    (SINT / ".pronto").write_text("ok")
    print(f"[sinais_pt] dataset sintético: {len(cods)} classes x ({n_treino}+{n_val}) -> {SINT}")
    return SINT


# ------------------------------ treino e avaliação ------------------------------
def treinar(dados, rapido=False):
    from ultralytics import YOLO
    runs = RAIZ / "va_datasets" / "runs"
    nome = "sinais_pt_cls_teste" if rapido else "sinais_pt_cls"
    m = YOLO(str(RAIZ / "modelos" / BASE) if (RAIZ / "modelos" / BASE).exists() else BASE)
    m.train(data=str(dados), epochs=1 if rapido else EPOCAS, imgsz=IMGSZ, batch=LOTE,
            project=str(runs), name=nome, exist_ok=True, patience=10,
            fliplr=0.0,                  # NUNCA espelhar
            erasing=0.2, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, workers=4, verbose=False, plots=False)
    return runs / nome / "weights" / "best.pt"


def avaliar(pesos, nomes):
    """Exatidão em FOTOGRAFIAS REAIS: recortes GTSRB (val) dos tipos que existem em PT."""
    from ultralytics import YOLO
    m = YOLO(str(pesos))
    cod_de = {}                                   # índice da classe -> código PT
    for i, n in m.names.items():
        cod_de[i] = n.split("_", 1)[1].split("__")[0] if "__" in n else n
    total = certos = 0
    por = {}
    for g, cod in GTSRB_PARA_PT.items():
        cod = cod.rstrip("_")
        if cod not in nomes:
            continue
        pasta = next(GTSRB_VAL.glob(f"{g:02d}_*"), None)
        if pasta is None:
            continue
        fich = sorted(pasta.glob("*.png"))
        imgs = [ler_imagem(f) for f in fich]
        imgs = [im for im in imgs if im is not None]
        res = m.predict(imgs, imgsz=IMGSZ, verbose=False, batch=128)
        ok = sum(cod_de[int(r.probs.top1)] == cod for r in res)
        por[cod] = (ok, len(res))
        total += len(res)
        certos += ok
    print("\n[sinais_pt] exatidão em fotografias reais (GTSRB -> código PT):")
    for cod, (ok, n) in sorted(por.items(), key=lambda kv: _ordem(kv[0])):
        print(f"   {cod:8s} {nomes[cod][:34]:34s} {100 * ok / max(n, 1):5.1f}%  ({n})")
    acc = certos / max(total, 1)
    print(f"   TOTAL {100 * acc:.1f}%  ({total} fotografias, {len(por)} tipos)")
    return acc, por


def escrever_nomes(nomes):
    """modelos/sinais_pt_nomes.json: código -> nome (o reconhecedor mostra 'B2 STOP'...)."""
    curto = {c: n.split(" (")[0] for c, n in nomes.items()}
    (RAIZ / "modelos" / "sinais_pt_nomes.json").write_text(
        json.dumps(curto, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-dados", action="store_true")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--so-avaliar", action="store_true")
    a = ap.parse_args()
    nomes = descarregar()
    dados = gerar(nomes)
    if a.so_dados:
        return
    pesos = RAIZ / "modelos" / "sinais_pt_cls.pt" if a.so_avaliar else treinar(dados, a.rapido)
    avaliar(pesos, nomes)
    if not a.rapido and not a.so_avaliar:
        shutil.copy2(pesos, RAIZ / "modelos" / "sinais_pt_cls.pt")
        escrever_nomes(nomes)
        print(f"[sinais_pt] instalado: modelos/sinais_pt_cls.pt ({len(nomes)} sinais PT)")


if __name__ == "__main__":
    main()
