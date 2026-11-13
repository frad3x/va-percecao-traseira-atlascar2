"""
gui.py — a APLICAÇÃO de janela: escolher a gravação, analisar, ver os resultados.

Abrir:  atalho "Perceção Traseira" no ambiente de trabalho
        ou  python -m percecao.gui   (a partir da pasta do projeto)

Três ecrãs:
  1) Início     — arrastar/escolher a gravação (vídeo, rosbag .mcap ou pasta) + opções
  2) A analisar — pré-visualização ao vivo, progresso, contagens e avisos
  3) Resultados — resumo em cartões + player, gráficos, pasta dos CSV, vídeo anotado
O processamento é o mesmo do main.py (percecao.pipeline), numa thread à parte.
"""
import collections
import datetime
import os
import sys
import time
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")
ICONE = os.path.join(RAIZ, "assets", "icone.ico")
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".mcap", ".m4v", ".wmv")

# Cores da Universidade de Aveiro (identidade de ua.pt): PRETO e BRANCO como base,
# VERDE UA (#92D400) como cor de destaque.
UA_VERDE = "#92D400"
UA_VERDE_HOVER = "#a6e21a"
UA_PRETO = "#000000"
ACCENT = UA_VERDE
QSS = f"""
* {{ font-family: 'Segoe UI', sans-serif; }}
QMainWindow, QWidget#raiz {{ background: #0f0f0f; }}
QWidget {{ color: #ececec; font-size: 13px; }}
QFrame#cabecalho {{ background: {UA_PRETO}; border-bottom: 4px solid {UA_VERDE}; }}
QLabel#marca {{ color: #ffffff; font-size: 12px; font-weight: 600; letter-spacing: 1px; }}
QLabel#titulo {{ font-size: 26px; font-weight: 700; color: #ffffff; }}
QLabel#subtitulo {{ color: #a3a3a3; font-size: 13px; }}
QLabel#seccao {{ color: {UA_VERDE}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#muted {{ color: #9a9a9a; }}
QFrame#cartao {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; }}
QFrame#zona {{ background: #151515; border: 2px dashed #3a3a3a; border-radius: 16px; }}
QFrame#zona[ativa="true"] {{ border-color: {UA_VERDE}; background: #1a2210; }}
QFrame#zona[escolhida="true"] {{ border-style: solid; border-color: {UA_VERDE}; background: #182010; }}
QPushButton {{ background: #262626; border: 1px solid #383838; border-radius: 9px;
              padding: 8px 16px; font-weight: 600; color: #f2f2f2; }}
QPushButton:hover {{ background: #303030; border-color: #4a4a4a; }}
QPushButton:pressed {{ background: #3a3a3a; }}
QPushButton:disabled {{ color: #5f5f5f; background: #1b1b1b; border-color: #262626; }}
QPushButton#primario {{ background: {UA_VERDE}; border: none; color: {UA_PRETO}; font-size: 15px;
                       font-weight: 700; padding: 12px 28px; border-radius: 11px; }}
QPushButton#primario:hover {{ background: {UA_VERDE_HOVER}; }}
QPushButton#primario:disabled {{ background: #2c3320; color: #6b7358; }}
QPushButton#perigo {{ background: #3a1d1d; border-color: #5a2a2a; color: #ff8a8a; }}
QPushButton#perigo:hover {{ background: #472424; }}
QPushButton#seg {{ background: #1f1f1f; border: 1px solid #333333; border-radius: 8px;
                  padding: 7px 14px; color: #b5b5b5; }}
QPushButton#seg:hover {{ border-color: #555555; }}
QPushButton#seg:checked {{ background: {UA_VERDE}; border-color: {UA_VERDE}; color: {UA_PRETO}; }}
QPushButton#recente {{ background: transparent; border: none; text-align: left; color: {UA_VERDE};
                      padding: 3px 4px; font-weight: 500; }}
QPushButton#recente:hover {{ color: #ffffff; text-decoration: underline; }}
QCheckBox {{ spacing: 8px; color: #d0d0d0; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid #4a4a4a;
                        background: #1f1f1f; }}
QCheckBox::indicator:checked {{ background: {UA_VERDE}; border-color: {UA_VERDE}; }}
QSpinBox {{ background: #1f1f1f; border: 1px solid #333333; border-radius: 8px; padding: 5px 8px;
           min-width: 110px; }}
QSpinBox:focus {{ border-color: {UA_VERDE}; }}
QProgressBar {{ background: #232323; border: none; border-radius: 6px; height: 12px;
               text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {UA_VERDE}; border-radius: 6px; }}
QLabel#preview {{ background: #000000; border: 1px solid #2a2a2a; border-radius: 12px; }}
QLabel#chipnum {{ font-size: 22px; font-weight: 700; color: #ffffff; }}
QLabel#chiplab {{ color: #9a9a9a; font-size: 11px; }}
QLabel#aviso {{ background: #3a2c0c; color: #f3b21b; border-radius: 8px; padding: 8px 12px;
               font-weight: 600; }}
QLabel#aviso[nivel="perigo"] {{ background: #3d1a14; color: #f0592a; }}
QLabel#aviso[nivel="ok"] {{ background: #1a240b; color: {UA_VERDE}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #1a1a1a; padding: 8px 16px; border-radius: 8px; margin-right: 4px; }}
QTabBar::tab:selected {{ background: {UA_VERDE}; color: {UA_PRETO}; }}
QScrollArea {{ border: none; background: transparent; }}
QToolTip {{ background: #000000; color: #ffffff; border: 1px solid {UA_VERDE}; }}
"""


# ============================================================================
# utilitários de interface
# ============================================================================
def rotulo(txt, nome=None, wrap=False):
    l = QtWidgets.QLabel(txt)
    if nome:
        l.setObjectName(nome)
    l.setWordWrap(wrap)
    return l


def cartao():
    f = QtWidgets.QFrame()
    f.setObjectName("cartao")
    return f


def sombra(w, raio=28, alfa=110):
    ef = QtWidgets.QGraphicsDropShadowEffect(w)
    ef.setBlurRadius(raio)
    ef.setOffset(0, 6)
    ef.setColor(QtGui.QColor(0, 0, 0, alfa))
    w.setGraphicsEffect(ef)


def repolir(w):
    w.style().unpolish(w)
    w.style().polish(w)


class Segmentado(QtWidgets.QWidget):
    """Botões de escolha única lado a lado (ex.: Rápida | Precisa)."""
    mudou = QtCore.Signal(str)

    def __init__(self, opcoes, atual):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.grupo = QtWidgets.QButtonGroup(self)
        self.botoes = {}
        for chave, txt in opcoes:
            b = QtWidgets.QPushButton(txt)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, c=chave: self.mudou.emit(c))
            self.grupo.addButton(b)
            self.botoes[chave] = b
            lay.addWidget(b)
        lay.addStretch(1)
        self.definir(atual)

    def valor(self):
        return next(k for k, b in self.botoes.items() if b.isChecked())

    def definir(self, chave):
        if chave in self.botoes:
            self.botoes[chave].setChecked(True)


class Chip(QtWidgets.QFrame):
    """Número grande + legenda (contagens ao vivo / resultados)."""

    def __init__(self, legenda, cor=None):
        super().__init__()
        self.setObjectName("cartao")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.num = rotulo("—", "chipnum")
        if cor:
            self.num.setStyleSheet(f"color:{cor};")
        lay.addWidget(self.num)
        self.lab = rotulo(legenda, "chiplab", wrap=True)
        lay.addWidget(self.lab)

    def definir(self, valor, legenda=None):
        self.num.setText(str(valor))
        if legenda is not None:
            self.lab.setText(legenda)


class ZonaFicheiro(QtWidgets.QFrame):
    """Zona para arrastar a gravação (ou clicar para escolher)."""
    escolhida = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("zona")
        self.setAcceptDrops(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(170)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(QtCore.Qt.AlignCenter)
        lay.setSpacing(6)
        self.icone = rotulo("🎞️")
        self.icone.setStyleSheet("font-size: 38px;")
        self.icone.setAlignment(QtCore.Qt.AlignCenter)
        self.texto = rotulo("Arrasta para aqui a gravação", None)
        self.texto.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.texto.setAlignment(QtCore.Qt.AlignCenter)
        self.detalhe = rotulo("vídeo (.mp4, .avi…), rosbag (.mcap) ou pasta de imagens", "muted")
        self.detalhe.setAlignment(QtCore.Qt.AlignCenter)
        for w in (self.icone, self.texto, self.detalhe):
            lay.addWidget(w)
        bts = QtWidgets.QHBoxLayout()
        bts.setAlignment(QtCore.Qt.AlignCenter)
        self.bt_fich = QtWidgets.QPushButton("Escolher ficheiro…")
        self.bt_pasta = QtWidgets.QPushButton("Escolher pasta…")
        for b in (self.bt_fich, self.bt_pasta):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            bts.addWidget(b)
        lay.addSpacing(6)
        lay.addLayout(bts)
        self.bt_fich.clicked.connect(self.escolher_ficheiro)
        self.bt_pasta.clicked.connect(self.escolher_pasta)

    def _inicio(self):
        p = os.path.join(RAIZ, "dados", "bags")
        return p if os.path.isdir(p) else RAIZ

    def escolher_ficheiro(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolhe a gravação", self._inicio(),
            "Gravações (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.mcap);;Todos os ficheiros (*)")
        if f:
            self.escolhida.emit(f)

    def escolher_pasta(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Escolhe a pasta (rosbag ou imagens)", self._inicio())
        if d:
            self.escolhida.emit(d)

    def mostrar(self, caminho):
        nome = os.path.basename(caminho.rstrip("/\\")) or caminho
        self.icone.setText("✅")
        self.texto.setText(nome)
        self.detalhe.setText(descrever_fonte(caminho))
        self.setProperty("escolhida", True)
        repolir(self)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.escolher_ficheiro()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("ativa", True)
            repolir(self)

    def dragLeaveEvent(self, e):
        self.setProperty("ativa", False)
        repolir(self)

    def dropEvent(self, e):
        self.setProperty("ativa", False)
        repolir(self)
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.escolhida.emit(urls[0])


def descrever_fonte(caminho):
    if os.path.isdir(caminho):
        fs = os.listdir(caminho)
        if any(f.endswith(".mcap") for f in fs):
            return "rosbag (pasta com .mcap)"
        n = sum(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for f in fs)
        return f"pasta com {n} imagens" if n else "pasta"
    tam = os.path.getsize(caminho) / 1e6 if os.path.isfile(caminho) else 0
    tipo = "rosbag" if caminho.lower().endswith(".mcap") else "vídeo"
    return f"{tipo} · {tam:.0f} MB"


def para_qimage(bgr, largura=None):
    import cv2
    if largura and bgr.shape[1] > largura:
        bgr = cv2.resize(bgr, (largura, int(bgr.shape[0] * largura / bgr.shape[1])),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


# ============================================================================
# processamento numa thread
# ============================================================================
class Trabalhador(QtCore.QThread):
    estado = QtCore.Signal(str)
    progresso = QtCore.Signal(int, int, float, object, object)   # k, n, fps, QImage|None, info
    terminado = QtCore.Signal(object)
    erro = QtCore.Signal(str)

    def __init__(self, caminho, opcoes):
        super().__init__()
        self.caminho, self.op = caminho, opcoes
        self._parar = False

    def parar(self):
        self._parar = True

    def run(self):
        try:
            self._run()
        except Exception:
            self.erro.emit(traceback.format_exc())

    def _run(self):
        from . import app, desenho, fontes
        from .pipeline import Config
        op = self.op
        if self.caminho.lower().endswith(".mcap") or (
                os.path.isdir(self.caminho) and any(f.endswith(".mcap") for f in os.listdir(self.caminho))):
            self.estado.emit("A preparar o rosbag (só demora na 1.ª vez)…")
        else:
            self.estado.emit("A abrir a gravação…")
        fonte = fontes.Fonte(self.caminho)

        pasta = app.nova_pasta_resultados(self.caminho)
        seg = {"traseira": op["camara"] == "traseira"}
        if op["vel_kmh"]:
            seg["vel_ego_kmh"] = float(op["vel_kmh"])
        rapida = op["qualidade"] == "rapida"
        cfg = Config(modelo="modelos/yolo11n.pt" if rapida else "modelos/yolo11m.pt",
                     imgsz=640 if rapida else 1280, seguranca_params=seg)
        self.estado.emit("A carregar os modelos de deteção…")

        t0 = time.time()
        ultimo_prev = [0.0]
        estacionamento = op["estacionamento"]

        def progresso(k, n, res, frame):
            agora = time.time()
            img = None
            if agora - ultimo_prev[0] > 0.25:          # pré-visualização ~4x/s
                ultimo_prev[0] = agora
                img = para_qimage(desenho.desenhar(frame, res, ttc_risco=cfg.limiar_risco,
                                                   estacionamento=estacionamento), 960)
            seg_ = res.get("seg") or {}
            info = dict(veic=len(res["veic"]["bb"]), duas=len(res["duas_rodas"]["bb"]),
                        peoes=len(res["peoes"]["bb"]), sinais=len(res["sinais"]["bb"]),
                        alertas=seg_.get("alertas", []), linhas=(res.get("linhas") or {}).get("n", 0))
            self.progresso.emit(k, n, (k + 1) / max(agora - t0, 1e-6), img, info)

        primeira = [True]

        def log(msg):
            if primeira[0] and msg.startswith("Sinais"):
                primeira[0] = False
                self.estado.emit("A analisar…")

        video = os.path.join(pasta, "video_anotado.mp4") if op["gravar_video"] else None
        resultados, stats = app.processar(
            fonte, cfg, progresso=progresso, parar=lambda: self._parar,
            gravar_video=bool(video), fich_video=video or "", max_frames=op["max_frames"],
            estacionamento=estacionamento, log=log)
        dt = time.time() - t0
        if not resultados:
            raise RuntimeError("A gravação não tem imagens que se consigam ler.")
        self.estado.emit("A guardar os resultados…")
        app.exportar(resultados, stats, cfg, pasta)
        app._graficos(stats, False, cfg.limiar_risco, pasta=pasta)
        self.terminado.emit(dict(fonte=fonte, resultados=resultados, stats=stats, pasta=pasta,
                                 video=video, dt=dt, cfg=cfg, cancelado=self._parar,
                                 caminho=self.caminho))


# ============================================================================
# janela principal
# ============================================================================
class Janela(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perceção Traseira — ATLASCAR2")
        if os.path.isfile(ICONE):
            self.setWindowIcon(QtGui.QIcon(ICONE))
        self.resize(1120, 800)
        self.cfg = QtCore.QSettings("G6-VA", "PercecaoTraseira")
        self.caminho = None
        self.trab = None
        self.res = None
        self._janelas = []                # player/gráficos abertos (não deixar o GC fechar)

        raiz = QtWidgets.QWidget()
        raiz.setObjectName("raiz")
        self.setCentralWidget(raiz)
        externo = QtWidgets.QVBoxLayout(raiz)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)

        # cabeçalho preto com a faixa verde (estilo UA)
        cabecalho = QtWidgets.QFrame()
        cabecalho.setObjectName("cabecalho")
        cab = QtWidgets.QHBoxLayout(cabecalho)
        cab.setContentsMargins(36, 18, 36, 18)
        tit = QtWidgets.QVBoxLayout()
        tit.setSpacing(2)
        tit.addWidget(rotulo("UNIVERSIDADE DE AVEIRO  ·  DEPARTAMENTO DE ENGENHARIA MECÂNICA", "marca"))
        tit.addWidget(rotulo("Perceção Traseira", "titulo"))
        tit.addWidget(rotulo("ATLASCAR2 · Veículos Autónomos · grupo G6 — veículos, peões, linhas, "
                             "sinais e avisos de segurança", "subtitulo"))
        cab.addLayout(tit, 1)
        self.lbl_gpu = rotulo("a verificar a GPU…", "muted")
        cab.addWidget(self.lbl_gpu, 0, QtCore.Qt.AlignTop)
        externo.addWidget(cabecalho)

        corpo = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(corpo)
        lay.setContentsMargins(36, 22, 36, 26)
        lay.setSpacing(18)
        externo.addWidget(corpo, 1)

        self.pilha = QtWidgets.QStackedWidget()
        lay.addWidget(self.pilha, 1)
        self.pilha.addWidget(self._pagina_inicio())
        self.pilha.addWidget(self._pagina_progresso())
        self.pilha.addWidget(self._pagina_resultados())
        QtCore.QTimer.singleShot(150, self._verificar_gpu)

    # ------------------------------------------------------------ página 1
    def _pagina_inicio(self):
        p = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(p)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        self.zona = ZonaFicheiro()
        self.zona.escolhida.connect(self._escolher)
        sombra(self.zona, 24, 70)
        lay.addWidget(self.zona)

        self.caixa_recentes = QtWidgets.QHBoxLayout()
        self.caixa_recentes.setSpacing(4)
        lay.addLayout(self.caixa_recentes)
        self._atualizar_recentes()

        op = cartao()
        sombra(op, 24, 70)
        g = QtWidgets.QGridLayout(op)
        g.setContentsMargins(22, 18, 22, 18)
        g.setHorizontalSpacing(26)
        g.setVerticalSpacing(14)
        g.addWidget(rotulo("OPÇÕES", "seccao"), 0, 0, 1, 4)

        self.seg_qual = Segmentado([("rapida", "⚡ Rápida"), ("precisa", "🎯 Precisa")],
                                   self.cfg.value("qualidade", "precisa"))
        self.lbl_qual = rotulo("", "muted", wrap=True)
        self.seg_qual.mudou.connect(lambda _: self._texto_qualidade())
        g.addWidget(rotulo("Qualidade"), 1, 0)
        g.addWidget(self.seg_qual, 1, 1)
        g.addWidget(self.lbl_qual, 2, 1)

        self.seg_cam = Segmentado([("traseira", "Traseira"), ("frontal", "Frontal")],
                                  self.cfg.value("camara", "traseira"))
        g.addWidget(rotulo("Câmara"), 1, 2)
        g.addWidget(self.seg_cam, 1, 3)
        g.addWidget(rotulo("frontal: vídeos de teste com a câmara virada para a frente", "muted",
                           wrap=True), 2, 3)

        self.sp_vel = QtWidgets.QSpinBox()
        self.sp_vel.setRange(0, 200)
        self.sp_vel.setSuffix(" km/h")
        self.sp_vel.setSpecialValueText("desconhecida")
        self.sp_vel.setValue(int(self.cfg.value("vel_kmh", 0)))
        g.addWidget(rotulo("Velocidade do carro"), 3, 0)
        g.addWidget(self.sp_vel, 3, 1, QtCore.Qt.AlignLeft)

        self.seg_dur = Segmentado([("tudo", "Tudo"), ("parte", "Só o início")],
                                  self.cfg.value("duracao", "tudo"))
        self.sp_frames = QtWidgets.QSpinBox()
        self.sp_frames.setRange(10, 100000)
        self.sp_frames.setSingleStep(100)
        self.sp_frames.setSuffix(" imagens")
        self.sp_frames.setValue(int(self.cfg.value("max_frames", 300)))
        self.seg_dur.mudou.connect(lambda c: self.sp_frames.setEnabled(c == "parte"))
        self.sp_frames.setEnabled(self.seg_dur.valor() == "parte")
        dur = QtWidgets.QHBoxLayout()
        dur.addWidget(self.seg_dur)
        dur.addWidget(self.sp_frames)
        dur.addStretch(1)
        g.addWidget(rotulo("Analisar"), 3, 2)
        g.addLayout(dur, 3, 3)

        self.ck_video = QtWidgets.QCheckBox("Gravar vídeo anotado (.mp4)")
        self.ck_video.setChecked(self.cfg.value("gravar_video", "false") == "true")
        self.ck_estac = QtWidgets.QCheckBox("Guias de estacionamento no vídeo")
        self.ck_estac.setChecked(self.cfg.value("estacionamento", "false") == "true")
        g.addWidget(self.ck_video, 4, 1)
        g.addWidget(self.ck_estac, 4, 3)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        lay.addWidget(op)
        self._texto_qualidade()

        lay.addStretch(1)
        fundo = QtWidgets.QHBoxLayout()
        fundo.addWidget(rotulo("Os resultados ficam em  Projeto\\resultados\\", "muted"))
        fundo.addStretch(1)
        self.bt_analisar = QtWidgets.QPushButton("Analisar  ▶")
        self.bt_analisar.setObjectName("primario")
        self.bt_analisar.setCursor(QtCore.Qt.PointingHandCursor)
        self.bt_analisar.setEnabled(False)
        self.bt_analisar.clicked.connect(self._analisar)
        sombra(self.bt_analisar, 22, 90)
        fundo.addWidget(self.bt_analisar)
        lay.addLayout(fundo)
        return p

    def _texto_qualidade(self):
        if self.seg_qual.valor() == "rapida":
            self.lbl_qual.setText("YOLO11n a 640 px — bom sem placa gráfica")
        else:
            self.lbl_qual.setText("YOLO11m a 1280 px — apanha objetos pequenos (precisa de GPU)")

    def _atualizar_recentes(self):
        while self.caixa_recentes.count():
            w = self.caixa_recentes.takeAt(0).widget()
            if w:
                w.deleteLater()
        rec = [r for r in (self.cfg.value("recentes", []) or []) if os.path.exists(r)][:4]
        if not rec:
            return
        self.caixa_recentes.addWidget(rotulo("Recentes:", "muted"))
        for r in rec:
            b = QtWidgets.QPushButton(os.path.basename(r.rstrip("/\\")))
            b.setObjectName("recente")
            b.setToolTip(r)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, c=r: self._escolher(c))
            self.caixa_recentes.addWidget(b)
        self.caixa_recentes.addStretch(1)

    def _escolher(self, caminho):
        ok = os.path.isdir(caminho) or caminho.lower().endswith(EXT_VIDEO)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Ficheiro não suportado",
                                          "Escolhe um vídeo, um rosbag (.mcap) ou uma pasta.")
            return
        self.caminho = caminho
        self.zona.mostrar(caminho)
        self.bt_analisar.setEnabled(True)

    def _verificar_gpu(self):
        class Gpu(QtCore.QThread):
            pronto = QtCore.Signal(str)

            def run(self):
                try:
                    import torch
                    self.pronto.emit(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
                except Exception:
                    self.pronto.emit("")
        self._gpu = Gpu()
        self._gpu.pronto.connect(self._gpu_pronta)
        self._gpu.start()

    def _gpu_pronta(self, nome):
        if nome:
            self.lbl_gpu.setText(f"🟢 GPU: {nome}")
        else:
            self.lbl_gpu.setText("⚪ sem GPU NVIDIA — usa a qualidade Rápida")
            if self.cfg.value("qualidade") is None:
                self.seg_qual.definir("rapida")
                self._texto_qualidade()

    # ------------------------------------------------------------ página 2
    def _pagina_progresso(self):
        p = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(p)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        topo = QtWidgets.QHBoxLayout()
        self.lbl_nome = rotulo("", None)
        self.lbl_nome.setStyleSheet("font-size: 17px; font-weight: 700;")
        topo.addWidget(self.lbl_nome, 1)
        self.lbl_estado = rotulo("", "muted")
        topo.addWidget(self.lbl_estado)
        lay.addLayout(topo)

        meio = QtWidgets.QHBoxLayout()
        meio.setSpacing(16)
        self.preview = QtWidgets.QLabel()
        self.preview.setObjectName("preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(560, 320)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._ultima_img = None
        meio.addWidget(self.preview, 1)

        lado = QtWidgets.QVBoxLayout()
        lado.setSpacing(10)
        self.chips = {}
        for chave, leg, cor in (("veic", "veículos", "#4aa8ff"), ("duas", "duas rodas", "#ff9a3c"),
                                ("peoes", "peões", "#ffe14d"), ("sinais", "sinais", "#e77be7"),
                                ("linhas", "linhas da faixa", "#92D400")):
            c = Chip(leg + " nesta imagem", cor)
            c.setFixedWidth(190)
            self.chips[chave] = c
            lado.addWidget(c)
        lado.addStretch(1)
        meio.addLayout(lado)
        lay.addLayout(meio, 1)

        self.lbl_aviso = rotulo("Sem avisos", "aviso")
        self.lbl_aviso.setProperty("nivel", "ok")
        lay.addWidget(self.lbl_aviso)

        self.barra = QtWidgets.QProgressBar()
        self.barra.setRange(0, 0)
        lay.addWidget(self.barra)
        fundo = QtWidgets.QHBoxLayout()
        self.lbl_prog = rotulo("", "muted")
        fundo.addWidget(self.lbl_prog, 1)
        self.bt_cancelar = QtWidgets.QPushButton("Parar e ver o que já foi analisado")
        self.bt_cancelar.setObjectName("perigo")
        self.bt_cancelar.setCursor(QtCore.Qt.PointingHandCursor)
        self.bt_cancelar.clicked.connect(self._cancelar)
        fundo.addWidget(self.bt_cancelar)
        lay.addLayout(fundo)
        return p

    def _analisar(self):
        if not self.caminho:
            return
        op = dict(qualidade=self.seg_qual.valor(), camara=self.seg_cam.valor(),
                  vel_kmh=self.sp_vel.value(),
                  max_frames=self.sp_frames.value() if self.seg_dur.valor() == "parte" else 0,
                  gravar_video=self.ck_video.isChecked(), estacionamento=self.ck_estac.isChecked())
        for k, v in (("qualidade", op["qualidade"]), ("camara", op["camara"]),
                     ("vel_kmh", op["vel_kmh"]), ("duracao", self.seg_dur.valor()),
                     ("max_frames", self.sp_frames.value()),
                     ("gravar_video", str(op["gravar_video"]).lower()),
                     ("estacionamento", str(op["estacionamento"]).lower())):
            self.cfg.setValue(k, v)
        rec = [self.caminho] + [r for r in (self.cfg.value("recentes", []) or []) if r != self.caminho]
        self.cfg.setValue("recentes", rec[:6])

        self._fechar_fonte()
        self.lbl_nome.setText(os.path.basename(self.caminho.rstrip("/\\")))
        self.lbl_estado.setText("A preparar…")
        self.barra.setRange(0, 0)
        self.lbl_prog.setText("")
        self.preview.clear()
        self.preview.setText("A preparar…")
        self._ultima_img = None
        self.bt_cancelar.setEnabled(True)
        self.pilha.setCurrentIndex(1)

        self.trab = Trabalhador(self.caminho, op)
        self.trab.estado.connect(self.lbl_estado.setText)
        self.trab.progresso.connect(self._progresso)
        self.trab.terminado.connect(self._terminado)
        self.trab.erro.connect(self._erro)
        self.trab.start()

    def _progresso(self, k, n, fps, img, info):
        if self.lbl_estado.text() != "A analisar…":
            self.lbl_estado.setText("A analisar…")
        if n:
            self.barra.setRange(0, n)
            self.barra.setValue(k + 1)
            falta = (n - k - 1) / max(fps, 1e-6)
            self.lbl_prog.setText(f"{k + 1} / {n} imagens  ·  {fps:.1f} imagens/s  ·  "
                                  f"faltam {_duracao(falta)}")
        else:
            self.lbl_prog.setText(f"{k + 1} imagens  ·  {fps:.1f} imagens/s")
        for c in ("veic", "duas", "peoes", "sinais", "linhas"):
            self.chips[c].definir(info[c])
        al = info["alertas"]
        if al:
            nivel, txt = al[0]
            self.lbl_aviso.setText(("⛔ " if nivel == "perigo" else "⚠️ ") + txt.capitalize())
            self.lbl_aviso.setProperty("nivel", nivel)
        else:
            self.lbl_aviso.setText("✓ Sem avisos")
            self.lbl_aviso.setProperty("nivel", "ok")
        repolir(self.lbl_aviso)
        if img is not None:
            self._ultima_img = img
            self._mostrar_preview()

    def _mostrar_preview(self):
        if self._ultima_img is not None:
            pix = QtGui.QPixmap.fromImage(self._ultima_img)
            self.preview.setPixmap(pix.scaled(self.preview.size() - QtCore.QSize(4, 4),
                                              QtCore.Qt.KeepAspectRatio,
                                              QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._mostrar_preview()

    def _cancelar(self):
        if self.trab is not None:
            self.trab.parar()
            self.bt_cancelar.setEnabled(False)
            self.lbl_estado.setText("A terminar…")

    def _erro(self, tb):
        self.pilha.setCurrentIndex(0)
        caixa = QtWidgets.QMessageBox(self)
        caixa.setIcon(QtWidgets.QMessageBox.Critical)
        caixa.setWindowTitle("Não foi possível analisar")
        ultima = tb.strip().splitlines()[-1]
        caixa.setText(f"Ocorreu um erro durante a análise:\n\n{ultima}")
        caixa.setDetailedText(tb)
        caixa.exec()

    # ------------------------------------------------------------ página 3
    def _pagina_resultados(self):
        p = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(p)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self.lbl_res_tit = rotulo("Análise concluída", None)
        self.lbl_res_tit.setStyleSheet("font-size: 19px; font-weight: 700;")
        lay.addWidget(self.lbl_res_tit)
        self.lbl_res_sub = rotulo("", "muted", wrap=True)
        lay.addWidget(self.lbl_res_sub)

        grelha = QtWidgets.QGridLayout()
        grelha.setSpacing(12)
        self.cards = {}
        defs = [("frames", "imagens analisadas", None), ("linhas", "com as 2 linhas da faixa", "#92D400"),
                ("veic", "veículos diferentes", "#4aa8ff"), ("duas", "duas rodas diferentes", "#ff9a3c"),
                ("peoes", "peões diferentes", "#ffe14d"), ("sinais", "tipos de sinal reconhecidos", "#e77be7"),
                ("avisos", "das imagens com avisos", "#ffb547"), ("ttc", "tempo até à colisão mínimo", "#ff7b86")]
        for i, (k, leg, cor) in enumerate(defs):
            c = Chip(leg, cor)
            sombra(c, 18, 60)
            self.cards[k] = c
            grelha.addWidget(c, i // 4, i % 4)
        lay.addLayout(grelha)

        det = cartao()
        dl = QtWidgets.QVBoxLayout(det)
        dl.setContentsMargins(20, 14, 20, 14)
        dl.addWidget(rotulo("DETALHES", "seccao"))
        self.lbl_det = rotulo("", None, wrap=True)
        self.lbl_det.setTextFormat(QtCore.Qt.RichText)
        self.lbl_det.setStyleSheet("line-height: 150%; color: #c9ceda;")
        dl.addWidget(self.lbl_det)
        lay.addWidget(det)
        lay.addStretch(1)

        bts = QtWidgets.QHBoxLayout()
        bts.setSpacing(10)
        self.bt_player = QtWidgets.QPushButton("▶  Abrir o player")
        self.bt_player.setObjectName("primario")
        self.bt_graf = QtWidgets.QPushButton("📈  Gráficos")
        self.bt_pasta = QtWidgets.QPushButton("📂  Pasta dos resultados")
        self.bt_video = QtWidgets.QPushButton("🎬  Vídeo anotado")
        self.bt_nova = QtWidgets.QPushButton("＋  Nova análise")
        for b in (self.bt_player, self.bt_graf, self.bt_pasta, self.bt_video):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            bts.addWidget(b)
        bts.addStretch(1)
        self.bt_nova.setCursor(QtCore.Qt.PointingHandCursor)
        bts.addWidget(self.bt_nova)
        sombra(self.bt_player, 22, 90)
        self.bt_player.clicked.connect(self._abrir_player)
        self.bt_graf.clicked.connect(self._abrir_graficos)
        self.bt_pasta.clicked.connect(lambda: _abrir(self.res["pasta"]))
        self.bt_video.clicked.connect(lambda: _abrir(self.res["video"]))
        self.bt_nova.clicked.connect(self._nova)
        lay.addLayout(bts)
        return p

    def _terminado(self, r):
        self.res = r
        st, rs = r["stats"], r["resultados"]
        n = len(st)
        ids = collections.defaultdict(set)
        sinais = set()
        for res in rs:
            for g in ("veic", "duas_rodas", "peoes"):
                ids[g].update(i for i in res[g].get("id", []) if i is not None and i >= 0)
            sinais.update(s["nome"] for s in res.get("sinais_vistos", []))
        com2 = sum(1 for s in st if s.get("nlinhas") == 2)
        com_av = sum(1 for s in st if s.get("alertas"))
        ttcs = [s["ttc_min"] for s in st if s["ttc_min"] != float("inf")]
        self.cards["frames"].definir(n)
        self.cards["linhas"].definir(f"{100 * com2 / n:.0f} %")
        self.cards["veic"].definir(len(ids["veic"]))
        self.cards["duas"].definir(len(ids["duas_rodas"]))
        self.cards["peoes"].definir(len(ids["peoes"]))
        self.cards["sinais"].definir(len(sinais))
        self.cards["avisos"].definir(f"{100 * com_av / n:.0f} %")
        self.cards["ttc"].definir(f"{min(ttcs):.1f} s" if ttcs else "—")

        avisos = collections.Counter()
        for s in st:
            for a in filter(None, (s.get("alertas") or "").split(" | ")):
                avisos[_tipo_aviso(a)] += 1
        est = collections.Counter(s.get("estado_dist") for s in st)
        tipos_e = [s.get("tipo_esq") for s in st if s.get("tipo_esq")]
        tipos_d = [s.get("tipo_dir") for s in st if s.get("tipo_dir")]
        linhas = []
        if sinais:
            linhas.append("<b>Sinais:</b> " + ", ".join(sorted(sinais)))
        if avisos:
            linhas.append("<b>Avisos mais frequentes:</b> " + " · ".join(
                f"{a} ({c} imagens)" for a, c in avisos.most_common(4)))
        if est.get("ok") or est.get("atencao") or est.get("perigo"):
            linhas.append("<b>Veículo atrás na nossa faixa:</b> " + " · ".join(
                f"{nome} {100 * est.get(k, 0) / n:.0f}%" for k, nome in
                (("ok", "a distância segura"), ("atencao", "perto demais"), ("perigo", "em risco"))))
        if tipos_e or tipos_d:
            linhas.append("<b>Tipo das linhas:</b> " + " · ".join(
                f"{lado}: {_maioria(t)}" for lado, t in (("esquerda (imagem)", tipos_e),
                                                         ("direita (imagem)", tipos_d)) if t))
        linhas.append(f"<span style='color:#8b93a3'>Ficheiros: deteccoes.csv, linhas.csv, "
                      f"seguranca.csv e gráficos em {r['pasta']}</span>")
        self.lbl_det.setText("<br>".join(linhas))

        fps = n / max(r["dt"], 1e-6)
        self.lbl_res_tit.setText("Análise interrompida — resultados parciais" if r["cancelado"]
                                 else "Análise concluída")
        nome = os.path.basename(r["caminho"].rstrip("/").rstrip(os.sep))
        self.lbl_res_sub.setText(f"{nome}  ·  {_duracao(r['dt'])} ({fps:.1f} imagens/s)")
        self.bt_video.setVisible(bool(r["video"]) and os.path.isfile(r["video"] or ""))
        self.pilha.setCurrentIndex(2)
        QtWidgets.QApplication.alert(self)

    def _abrir_player(self):
        from . import player
        r = self.res
        w = player.reproduzir(r["fonte"], r["resultados"], r["stats"], fps=r["fonte"].fps,
                              ttc_risco=r["cfg"].limiar_risco, bloquear=False)
        if w is not None:
            if os.path.isfile(ICONE):
                w.setWindowIcon(QtGui.QIcon(ICONE))
            self._janelas.append(w)

    def _abrir_graficos(self):
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Gráficos")
        d.resize(980, 720)
        tabs = QtWidgets.QTabWidget()
        for fich, nome in (("graf_deteccoes.png", "Deteções por imagem"),
                           ("graf_distancia.png", "Distância ao veículo de trás"),
                           ("graf_ttc.png", "Tempo até à colisão")):
            p = os.path.join(self.res["pasta"], fich)
            if os.path.isfile(p):
                l = QtWidgets.QLabel()
                l.setAlignment(QtCore.Qt.AlignCenter)
                l.setPixmap(QtGui.QPixmap(p).scaled(940, 640, QtCore.Qt.KeepAspectRatio,
                                                    QtCore.Qt.SmoothTransformation))
                l.setStyleSheet("background: white; border-radius: 10px;")
                tabs.addTab(l, nome)
        QtWidgets.QVBoxLayout(d).addWidget(tabs)
        d.show()
        self._janelas.append(d)

    def _nova(self):
        self.pilha.setCurrentIndex(0)
        self._atualizar_recentes()

    def _fechar_fonte(self):
        if self.res and self.res.get("fonte"):
            try:
                self.res["fonte"].fechar()
            except Exception:
                pass
        self.res = None

    def closeEvent(self, e):
        if self.trab is not None and self.trab.isRunning():
            self.trab.parar()
            self.trab.wait(3000)
        self._fechar_fonte()
        for w in self._janelas:
            try:
                w.close()
            except Exception:
                pass
        super().closeEvent(e)


# ============================================================================
def _duracao(s):
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def _maioria(lista):
    c = collections.Counter(lista).most_common(1)[0]
    return f"{c[0]} ({100 * c[1] / len(lista):.0f}%)"


def _tipo_aviso(txt):
    """'VEICULO ATRAS A 10 m (< 31 m)' -> 'Veículo atrás perto demais' (agrupa por tipo)."""
    t = txt.upper()
    for chave, nome in (("COLISAO", "Risco de colisão traseira"), ("VEICULO ATRAS", "Veículo atrás perto demais"),
                        ("ANGULO MORTO", "Veículo no ângulo morto"), ("DUAS RODAS", "Duas rodas a aproximar-se"),
                        ("PEAO", "Peão atrás")):
        if chave in t:
            return nome
    return txt.capitalize()


def _abrir(caminho):
    if caminho and os.path.exists(caminho):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(caminho))


def _excecao(tipo, valor, tb):
    texto = "".join(traceback.format_exception(tipo, valor, tb))
    try:
        os.makedirs(PASTA_RESULTADOS, exist_ok=True)
        with open(os.path.join(PASTA_RESULTADOS, "erros.log"), "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.datetime.now()} ===\n{texto}")
    except Exception:
        pass
    if QtWidgets.QApplication.instance():
        QtWidgets.QMessageBox.critical(None, "Erro inesperado", f"{valor}\n\n(ver resultados/erros.log)")


def main():
    os.chdir(RAIZ)                         # modelos/ e pesos YOLO relativos ao projeto
    sys.excepthook = _excecao
    if sys.platform == "win32":            # ícone próprio na barra de tarefas
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("G6.VA.PercecaoTraseira")
        except Exception:
            pass
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Perceção Traseira")
    app.setStyleSheet(QSS)
    if os.path.isfile(ICONE):
        app.setWindowIcon(QtGui.QIcon(ICONE))
    j = Janela()
    j.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):   # abrir com um ficheiro (arrastar p/ o atalho)
        j._escolher(os.path.abspath(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
