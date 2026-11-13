"""
pipeline.py — o processamento de UMA imagem, de ponta a ponta.

É o núcleo partilhado por:
  - a aplicação off-line (main.py: vídeo / pasta / rosbag -> player, CSV, gráficos)
  - o nó ROS 2 (ros2_aquisicao/src/va_percecao: subscreve a câmara, publica deteções)

    pipe = Pipeline(Config(...))
    res  = pipe.processar(frame_bgr, t)      # t em segundos (tempo da imagem)
    linha_csv = pipe.estatisticas(res, idx)  # resumo dessa frame

res é um dict com os grupos 'veic', 'duas_rodas', 'peoes', 'sinais' (caixas, IDs,
distância, faixa, TTC), 'linhas' (faixa + marcações no pavimento), 'seg' (avisos
de segurança) e 'sinais_vistos' (memória dos últimos sinais).
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    modelo: str = "modelos/yolo11m.pt"       # YOLO (veículos/duas rodas/peões)
    imgsz: int = 1280
    conf: float = 0.25
    usar_track: bool = True          # ByteTrack (IDs estáveis) — preciso para o TTC
    linhas: bool = True
    linhas_params: dict = field(default_factory=dict)
    sinais_det: str = "modelos/sinais_det.pt"
    sinais_cls: str = "modelos/sinais_cls.pt"
    sinais_classico: bool = True     # se não houver modelos treinados
    ttc: bool = True
    limiar_risco: float = 4.0        # TTC (s) abaixo do qual há risco
    seguranca: bool = True
    seguranca_params: dict = field(default_factory=dict)
    dispositivo: str = None          # None = automático (GPU se houver); "cpu", "0", ...


class Pipeline:
    def __init__(self, cfg=None, log=print):
        from . import detecao, aproximacao, desenho
        from .linhas import DetetorLinhas
        from .seguranca import AnaliseSeguranca

        self.cfg = cfg = cfg or Config()
        self.log = log
        self.det = detecao.Detetor(cfg.modelo, conf=cfg.conf, imgsz=cfg.imgsz,
                                   usar_track=cfg.usar_track, dispositivo=cfg.dispositivo)
        self.det_linhas = DetetorLinhas(cfg.linhas_params) if cfg.linhas else None

        self.rec_sinais = None
        if cfg.sinais_det and os.path.isfile(cfg.sinais_det):
            from .sinais_treinados import ReconhecedorSinais
            self.rec_sinais = ReconhecedorSinais(cfg.sinais_det, cfg.sinais_cls, imgsz=cfg.imgsz)
            log("Sinais: modelos TREINADOS (deteção GTSDB + 43 tipos GTSRB)")
        elif cfg.sinais_classico:
            log("Sinais: YOLO + detetor clássico (cor + forma)")

        self.aprox = aproximacao.Aproximacao(limiar_risco=cfg.limiar_risco) if cfg.ttc else None
        self.seguranca = AnaliseSeguranca(dict(cfg.seguranca_params, ttc_risco=cfg.limiar_risco),
                                          cfg.linhas_params) if cfg.seguranca else None
        self.memoria_sinais = desenho.MemoriaSinais()

    # ------------------------------------------------------------------
    def velocidade(self, v_ms):
        """Velocidade do carro (m/s), se conhecida -> regra dos 2 segundos."""
        if self.seguranca is not None:
            self.seguranca.p["vel_ego_kmh"] = None if v_ms is None else 3.6 * float(v_ms)

    def processar(self, frame, t):
        from . import sinais as msinais
        cfg = self.cfg
        res = self.det.detetar(frame)
        if self.rec_sinais is not None:
            res["sinais"] = msinais.combinar(res["sinais"], self.rec_sinais.detetar(frame))
        elif cfg.sinais_classico:
            res["sinais"] = msinais.combinar(res["sinais"], msinais.detetar(frame))
        if self.det_linhas is not None:
            # as caixas dos agentes não são linhas
            res["linhas"] = self.det_linhas.detetar(
                frame, excluir=res["veic"]["bb"] + res["duas_rodas"]["bb"] + res["peoes"]["bb"])

        res["ttc_min"] = float("inf")
        if self.aprox is not None:
            for g in ("veic", "duas_rodas"):
                res[g]["ttc"] = self.aprox.atualizar(res[g], t, g)
            res["ttc_min"] = self.aprox.ttc_minimo(res["veic"]["ttc"] + res["duas_rodas"]["ttc"])
        if self.seguranca is not None:
            self.seguranca.analisar(frame.shape, res, t)
        res["sinais_vistos"] = self.memoria_sinais.atualizar(frame, res["sinais"], t)
        res["t"] = t
        return res

    def estatisticas(self, res, idx):
        """Uma linha de resumo da frame (para CSV/gráficos)."""
        seg = res.get("seg") or {}
        seguidor = seg.get("seguidor") or {}
        mud = seg.get("mudanca", {})
        lin = res.get("linhas") or {}
        ttc_min = res.get("ttc_min", float("inf"))
        return dict(frame=idx,
                    t=res.get("t", 0.0),
                    nlinhas=lin.get("n", 0),
                    tipo_esq=(lin.get("esq") or {}).get("tipo"),     # aula 5, ex. 4a
                    tipo_dir=(lin.get("dir") or {}).get("tipo"),
                    marcacoes="+".join(m["tipo"] for m in lin.get("marcacoes", [])),
                    nveic=len(res["veic"]["bb"]),
                    nduas=len(res["duas_rodas"]["bb"]),
                    npeoes=len(res["peoes"]["bb"]),
                    nsinais=len(res["sinais"]["bb"]),
                    ttc_min=ttc_min,
                    risco=bool(ttc_min < self.cfg.limiar_risco),
                    dist_atras=seguidor.get("d"),
                    estado_dist=seg.get("estado_dist"),
                    mudar_esq=mud.get("esquerda", (None,))[0],
                    mudar_dir=mud.get("direita", (None,))[0],
                    livre_atras=seg.get("livre_atras"),
                    alertas=" | ".join(a for _, a in seg.get("alertas", [])))
