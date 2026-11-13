#!/usr/bin/env python3
"""
criar_atalho.py — cria o ícone da aplicação e um atalho "Perceção Traseira" no
ambiente de trabalho (Windows). Corre UMA vez, a partir da pasta do projeto:

    python\\.venv\\Scripts\\python.exe ferramentas\\criar_atalho.py

O atalho abre a aplicação de janela (ferramentas/abrir_app.pyw -> percecao/gui.py) sem consola.
Também se pode arrastar um vídeo/bag para cima do atalho para o abrir logo.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(RAIZ, "assets")
VERDE = (146, 212, 0)       # verde UA (#92D400)


def criar_icone():
    """Quadrado preto (UA) com uma 'lente' branca e ondas de deteção a verde."""
    os.makedirs(ASSETS, exist_ok=True)
    N = 512
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([16, 16, N - 16, N - 16], radius=110, fill=(0, 0, 0, 255))
    d.rectangle([16, N - 120, N - 16, N - 96], fill=VERDE)          # faixa verde
    cx, cy = N // 2, 230
    for i, r in enumerate((190, 140, 90)):                           # ondas de deteção
        d.arc([cx - r, cy - r, cx + r, cy + r], start=200, end=340, fill=VERDE, width=22 - 4 * i)
    d.ellipse([cx - 62, cy - 62, cx + 62, cy + 62], fill=(255, 255, 255, 255))   # lente
    d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(0, 0, 0, 255))
    d.ellipse([cx - 12, cy - 22, cx + 4, cy - 6], fill=VERDE)
    d.rounded_rectangle([cx - 130, N - 78, cx + 130, N - 46], radius=16, fill=(255, 255, 255, 255))
    png = os.path.join(ASSETS, "icone.png")
    ico = os.path.join(ASSETS, "icone.ico")
    im.save(png)
    im.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def pythonw_base():
    """O pythonw.exe do Python BASE (o do venv criado pelo uv é um programa de consola,
    que abriria uma janela preta atrás da app)."""
    cfg = os.path.join(RAIZ, "python", ".venv", "pyvenv.cfg")
    if os.path.isfile(cfg):
        for linha in open(cfg, encoding="utf-8"):
            if linha.startswith("home"):
                p = os.path.join(linha.split("=", 1)[1].strip(), "pythonw.exe")
                if os.path.isfile(p):
                    return p
    p = os.path.join(sys.base_prefix, "pythonw.exe")
    return p if os.path.isfile(p) else sys.executable


def criar_atalho(ico):
    pythonw = pythonw_base()
    lancador = os.path.join(RAIZ, "ferramentas", "abrir_app.pyw")
    ps = f"""
$d = [Environment]::GetFolderPath('Desktop')
$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'Perceção Traseira.lnk'))
$s.TargetPath = '{pythonw}'
$s.Arguments = '"{lancador}"'
$s.WorkingDirectory = '{RAIZ}'
$s.IconLocation = '{ico}'
$s.Description = 'Perceção com a câmara traseira do ATLASCAR2 (grupo G6)'
$s.Save()
Write-Output (Join-Path $d 'Perceção Traseira.lnk')
"""
    # O PowerShell 5 lê scripts com acentos corretamente só em UTF-8 COM BOM (e via -File)
    import tempfile
    f = os.path.join(tempfile.gettempdir(), "va_criar_atalho.ps1")
    with open(f, "w", encoding="utf-8-sig") as fh:
        fh.write(ps)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f],
                       capture_output=True)
    os.remove(f)
    sai = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    if r.returncode != 0:
        raise SystemExit(sai)
    return sai.strip()


if __name__ == "__main__":
    ico = criar_icone()
    print("Ícone:", ico)
    if sys.platform == "win32":
        print("Atalho:", criar_atalho(ico))
