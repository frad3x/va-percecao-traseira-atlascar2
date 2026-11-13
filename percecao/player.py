"""
player.py — player interativo para rever o vídeo COM as deteções.

Interface principal em Qt (PySide6): janela com tema escuro, barra de transporte
(início / recuar / play-pausa / avançar / fim), barra de posição, seletor de
velocidade e painel de estatísticas (V/P/S e TTC com cor de risco). Se o PySide6
não estiver disponível, recai para um player simples em OpenCV.

Controlos (Qt e OpenCV):
  ESPAÇO play/pausa · A/D ou ←/→ recuar/avançar · +/- velocidade ·
  HOME/END início/fim · Q/ESC sair.

Como as deteções já estão calculadas, recuar/saltar é instantâneo.
"""
from . import desenho


def reproduzir(fonte, resultados, stats=None, fps=15.0, ttc_risco=4.0):
    if not resultados:
        print("Sem frames para mostrar."); return
    try:
        _reproduzir_qt(fonte, resultados, stats, fps, ttc_risco)
    except ImportError:
        print("(PySide6 indisponível — a usar o player simples em OpenCV)")
        _reproduzir_cv2(fonte, resultados, stats, fps, ttc_risco)


# ============================ player Qt (principal) ============================
def _reproduzir_qt(fonte, resultados, stats, fps, ttc_risco):
    import cv2
    import numpy as np
    from PySide6 import QtCore, QtGui, QtWidgets

    if not fps or fps <= 0:
        fps = 15.0
    n = len(resultados)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Player(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Perceção — câmara traseira (ATLASCAR2)")
            self.resize(1180, 820)
            self.k = 0
            self.playing = False
            self.vel = 1.0
            self._sync = False
            self._ultimo_rgb = None

            # --- vídeo ---
            self.lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            self.lbl.setMinimumSize(640, 360)
            self.lbl.setObjectName("video")

            # --- barra de estado (contagens + TTC) ---
            self.lblInfo = QtWidgets.QLabel("—")
            self.lblInfo.setObjectName("info")
            self.lblTTC = QtWidgets.QLabel("")
            self.lblTTC.setObjectName("ttc")

            # --- transporte ---
            self.btIni = _botao("⏮", "Início (Home)", self.ir_inicio)
            self.btRec = _botao("◀", "Recuar (A / ←)", lambda: self.passo(-1))
            self.btPlay = _botao("▶", "Play / Pausa (Espaço)", self.play_pausa, largo=True)
            self.btAva = _botao("▶", "Avançar (D / →)", lambda: self.passo(1))
            self.btFim = _botao("⏭", "Fim (End)", self.ir_fim)

            self.cboVel = QtWidgets.QComboBox()
            self.cboVel.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
            self.cboVel.setCurrentText("1×")
            self.cboVel.currentTextChanged.connect(self.mudar_vel)

            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, max(n - 1, 1))
            self.slider.valueChanged.connect(self.on_slider)

            self.lblPos = QtWidgets.QLabel("0 / 0")
            self.lblPos.setObjectName("pos")

            # --- layout ---
            barra = QtWidgets.QHBoxLayout()
            for b in (self.btIni, self.btRec, self.btPlay, self.btAva, self.btFim):
                barra.addWidget(b)
            barra.addSpacing(8)
            barra.addWidget(QtWidgets.QLabel("Veloc."))
            barra.addWidget(self.cboVel)
            barra.addSpacing(12)
            barra.addWidget(self.slider, 1)
            barra.addWidget(self.lblPos)

            estado = QtWidgets.QHBoxLayout()
            estado.addWidget(self.lblInfo)
            estado.addStretch(1)
            estado.addWidget(self.lblTTC)

            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(8)
            lay.addWidget(self.lbl, 1)
            lay.addLayout(estado)
            lay.addLayout(barra)

            self.setStyleSheet(_QSS)

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)

            self.render()

        # ---- render ----
        def render(self):
            self.k = max(0, min(self.k, n - 1))
            frame = fonte.ler(self.k)
            if frame is not None:
                out = desenho.desenhar(frame, resultados[self.k], ttc_risco=ttc_risco)
                self._ultimo_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            self._mostrar()
            self._sync = True
            self.slider.setValue(self.k)
            self._sync = False
            self._atualizar_info()

        def _mostrar(self):
            if self._ultimo_rgb is None:
                return
            rgb = np.ascontiguousarray(self._ultimo_rgb)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg.copy())
            self.lbl.setPixmap(pix.scaled(self.lbl.size(), QtCore.Qt.KeepAspectRatio,
                                          QtCore.Qt.SmoothTransformation))

        def _atualizar_info(self):
            r = resultados[self.k]
            nv = len(r.get("veic", {}).get("bb", []))
            npe = len(r.get("peoes", {}).get("bb", []))
            ns = len(r.get("sinais", {}).get("bb", []))
            lin = r.get("linhas") or {}
            nl = lin.get("n", 0)
            tps = [f"{s}: {(lin.get(k) or {}).get('tipo')}" for k, s in (("esq", "esq"), ("dir", "dir"))
                   if lin.get(k)]
            nl = f"{nl} ({', '.join(tps)})" if tps else nl
            t = ""
            if stats is not None and self.k < len(stats):
                tt = stats[self.k].get("ttc_min", float("inf"))
                if tt != float("inf"):
                    t = tt
            self.lblInfo.setText(
                f"<b>Linhas</b> {nl}&nbsp;&nbsp; <b>Veículos</b> {nv}&nbsp;&nbsp; "
                f"<b>Peões</b> {npe}&nbsp;&nbsp; <b>Sinais</b> {ns}")
            if t == "":
                self.lblTTC.setText("")
                self.lblTTC.setStyleSheet("")
            else:
                risco = t < ttc_risco
                cor = "#ff4d4d" if risco else "#36d399"
                aviso = "  ⚠ RISCO" if risco else ""
                self.lblTTC.setText(f"TTC {t:.1f}s{aviso}")
                self.lblTTC.setStyleSheet(f"color:{cor}; font-weight:700;")
            self.lblPos.setText(f"{self.k + 1} / {n}")

        # ---- controlo ----
        def play_pausa(self):
            self.playing = not self.playing
            self.btPlay.setText("⏸" if self.playing else "▶")
            if self.playing:
                if self.k >= n - 1:
                    self.k = 0
                self.timer.start(self._intervalo())
            else:
                self.timer.stop()

        def _intervalo(self):
            return max(int(1000.0 / (fps * self.vel)), 1)

        def _tick(self):
            if self.k >= n - 1:
                self.play_pausa()
                return
            self.k += 1
            self.render()

        def passo(self, d):
            if self.playing:
                self.play_pausa()
            self.k += d
            self.render()

        def ir_inicio(self):
            if self.playing:
                self.play_pausa()
            self.k = 0; self.render()

        def ir_fim(self):
            if self.playing:
                self.play_pausa()
            self.k = n - 1; self.render()

        def mudar_vel(self, txt):
            try:
                self.vel = float(txt.replace("×", ""))
            except ValueError:
                self.vel = 1.0
            if self.playing:
                self.timer.start(self._intervalo())

        def on_slider(self, v):
            if self._sync:
                return
            if self.playing:
                self.play_pausa()
            self.k = v
            self.render()

        def resizeEvent(self, e):
            self._mostrar()
            super().resizeEvent(e)

        def keyPressEvent(self, e):
            K = QtCore.Qt
            if e.key() == K.Key_Space:
                self.play_pausa()
            elif e.key() in (K.Key_Right, K.Key_D):
                self.passo(1)
            elif e.key() in (K.Key_Left, K.Key_A):
                self.passo(-1)
            elif e.key() in (K.Key_Plus, K.Key_Equal):
                self._mudar_vel_passo(+1)
            elif e.key() == K.Key_Minus:
                self._mudar_vel_passo(-1)
            elif e.key() == K.Key_Home:
                self.ir_inicio()
            elif e.key() == K.Key_End:
                self.ir_fim()
            elif e.key() in (K.Key_Q, K.Key_Escape):
                self.close()
            else:
                super().keyPressEvent(e)

        def _mudar_vel_passo(self, d):
            i = min(max(self.cboVel.currentIndex() + d, 0), self.cboVel.count() - 1)
            self.cboVel.setCurrentIndex(i)

    w = Player()
    w.show()
    app.exec()


def _botao(txt, dica, slot, largo=False):
    from PySide6 import QtCore, QtWidgets
    b = QtWidgets.QPushButton(txt)
    b.setToolTip(dica)
    b.setCursor(QtCore.Qt.PointingHandCursor)
    b.clicked.connect(slot)
    b.setFixedSize(64 if largo else 46, 38)
    b.setFocusPolicy(QtCore.Qt.NoFocus)
    return b


_QSS = """
QWidget { background:#15171c; color:#e6e8ee; font-family:'Segoe UI',sans-serif; font-size:13px; }
QLabel#video { background:#0b0c10; border:1px solid #2a2e37; border-radius:8px; }
QLabel#info { font-size:14px; }
QLabel#pos  { color:#9aa3b2; min-width:72px; }
QPushButton { background:#232732; border:1px solid #333948; border-radius:8px; font-size:16px; }
QPushButton:hover { background:#2d3340; }
QPushButton:pressed { background:#3a4252; }
QComboBox { background:#232732; border:1px solid #333948; border-radius:6px; padding:3px 8px; }
QComboBox QAbstractItemView { background:#232732; selection-background-color:#3a4252; }
QSlider::groove:horizontal { height:6px; background:#2a2e37; border-radius:3px; }
QSlider::sub-page:horizontal { background:#4c8bf5; border-radius:3px; }
QSlider::handle:horizontal { background:#eaecf2; width:14px; margin:-5px 0; border-radius:7px; }
QSlider::handle:horizontal:hover { background:#ffffff; }
"""


# ======================== player OpenCV (fallback) ========================
def _reproduzir_cv2(fonte, resultados, stats, fps, ttc_risco):
    import cv2
    n = len(resultados)
    if not fps or fps <= 0:
        fps = 15.0
    win = "Percecao - player (ESPACO play | a/d frame | +/- veloc | q sair)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(fonte.largura, 1280), min(fonte.altura, 720))
    est = {"k": 0, "play": False, "vel": 1.0}

    def on_tb(v):
        est["k"] = max(0, min(v, n - 1))
    cv2.createTrackbar("frame", win, 0, max(n - 1, 1), on_tb)

    while True:
        k = est["k"]
        frame = fonte.ler(k)
        if frame is not None:
            out = desenho.desenhar(frame, resultados[k], ttc_risco)
            cv2.imshow(win, out)
        if cv2.getTrackbarPos("frame", win) != k:
            cv2.setTrackbarPos("frame", win, k)
        espera = max(int(1000.0 / (fps * est["vel"])), 1) if est["play"] else 30
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if tecla == ord(" "):
            est["play"] = not est["play"]
        elif tecla in (ord("d"), 83):
            est["play"] = False; est["k"] = min(k + 1, n - 1)
        elif tecla in (ord("a"), 81):
            est["play"] = False; est["k"] = max(k - 1, 0)
        elif tecla in (ord("+"), ord("=")):
            est["vel"] = min(est["vel"] * 2, 8.0)
        elif tecla == ord("-"):
            est["vel"] = max(est["vel"] / 2, 0.25)
        elif est["play"]:
            est["k"] = k + 1
        if est["k"] >= n:
            est["play"] = False; est["k"] = n - 1
    cv2.destroyWindow(win)
