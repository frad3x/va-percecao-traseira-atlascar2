"""
Lança a aplicação de janela SEM consola (usado pelo atalho "Perceção Traseira").

Porquê este ficheiro: o ambiente do projeto (python/.venv, criado com o uv) tem um
pythonw.exe que é, na verdade, um programa de CONSOLA — abria uma janela preta atrás
da app. O atalho usa o pythonw.exe "verdadeiro" do Python base e este ficheiro junta-lhe
os pacotes do ambiente do projeto (YOLO, PySide6, ...).
"""
import os
import site
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pasta do projeto
SITE = os.path.join(RAIZ, "python", ".venv", "Lib", "site-packages")

# 1) pacotes do ambiente do projeto, com prioridade
if os.path.isdir(SITE):
    antes = list(sys.path)
    site.addsitedir(SITE)                 # também processa os .pth dos pacotes
    novos = [p for p in sys.path if p not in antes]
    sys.path[:] = [RAIZ] + novos + [p for p in antes if p != RAIZ]
else:
    sys.path.insert(0, RAIZ)

# 2) sem consola não há stdout/stderr: o texto vai para um ficheiro de registo
os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
_log = open(os.path.join(RAIZ, "resultados", "app.log"), "a", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = _log
if sys.stderr is None:
    sys.stderr = _log

# 3) programas auxiliares que as bibliotecas chamem (ex. nvidia-smi) também sem janela
if os.name == "nt":
    _Popen = subprocess.Popen

    class _PopenSemJanela(_Popen):
        def __init__(self, *a, **kw):
            if not kw.get("creationflags"):
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            super().__init__(*a, **kw)

    subprocess.Popen = _PopenSemJanela

from percecao.gui import main  # noqa: E402

main()
