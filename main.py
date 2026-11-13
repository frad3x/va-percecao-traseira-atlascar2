#!/usr/bin/env python3
"""
main.py — Perceção com câmara traseira (ATLASCAR2 / Tema 2). Versão SÓ Python.

Basta dar Run (ou: python main.py). Faz, numa só passagem sobre a fonte:
  - deteção + rastreio (YOLO) de veículos / duas rodas / peões / sinais
  - marcações no pavimento: linhas da faixa (+ tipo), passadeiras, linhas de paragem
  - sinais (modelos treinados, 43 tipos; ou clássico cor + forma)
  - tempo-para-colisão (TTC) dos veículos que seguem atrás
  - distância (monocular) a cada objeto e AVISOS: distância de segurança,
    mudança de faixa (faixa ao lado livre? linha contínua?), ângulo morto,
    peão atrás, espaço livre atrás (estacionamento)
  - estatísticas, gráficos e exportação para CSV
  - player interativo no fim (recuar, avançar, velocidade, seek)
Para o apoio ao estacionamento (3D/RGB-D) ver percecao/percecao3d.py.

Edita o bloco CONFIG abaixo e corre. (Também aceita variáveis de ambiente
VA_FONTE, VA_MAX_FRAMES, VA_MOSTRAR, VA_GRAFICOS — úteis para testes rápidos.)
"""
import json
import os
import time

# ========================= CONFIG (edita e dá Run) =========================
# >>> METE AQUI o caminho do que queres analisar (entre as aspas, depois do r):
FONTE = r"C:\Users\tomas\OneDrive\Documentos\Universidade\5º ano\VA\Projeto\dados\bags\vídeos teste\videoplayback.mp4"
#   -> vídeo .avi/.mp4, pasta de imagens, OU pasta/.mcap de um rosbag (converte sozinho)
#   Exemplos (Windows):
#     FONTE = r"C:\Users\tomas\Desktop\o_meu_video.mp4"
#     FONTE = r"dados\bags\g02_camtras_20261104_160818_percurso1"   # pasta do bag
FONTE = os.environ.get("VA_FONTE", FONTE)   # (permite também definir por variável de ambiente)
MODELO = "yolo11m.pt"      # modelo YOLO (descarregado na 1.ª vez). 'm' = bom equilíbrio
                           #   precisão/velocidade com GPU. Sem GPU: "yolo11n.pt" + IMGSZ=640.
IMGSZ = 1280               # resolução de inferência: 1280 apanha objetos PEQUENOS
                           #   (sinais ao longe) muito melhor que 640 — testado: 16x mais STOP
CONF = 0.25                # confiança mínima do YOLO
USAR_TRACK = True          # rastreio com IDs (ByteTrack) — preciso para o TTC
LINHAS = True              # deteção das linhas da faixa (método da aula 5, sem calibração)
# Afinar as linhas para ESTE vídeo (opcional; ver PARAMS em percecao/linhas.py). Ex.:
#   {"v_base": 0.82}     -> a câmara apanha o carro/capô em baixo (ex. Argoverse)
#   {"horizonte": 0.50}  -> fixar o horizonte em vez de o estimar
LINHAS_PARAMS = {}
SINAIS_CLASSICO = True     # YOLO (stop) + detetor clássico (restantes sinais PT)
# Sinais TREINADOS (gerados por treinar_sinais.py). Se existirem, substituem o
# detetor clássico e reconhecem 43 tipos (limite 50, cedência, rotunda, obras...).
SINAIS_DET = "modelos/sinais_det.pt"
SINAIS_CLS = "modelos/sinais_cls.pt"
ANALISAR_APROX = True      # TTC dos veículos atrás
LIMIAR_RISCO = 4.0         # TTC (s) abaixo do qual se assinala RISCO
SEGURANCA = True           # distâncias + avisos (ver percecao/seguranca.py)
SEGURANCA_PARAMS = {}      # ex.: {"vel_ego_kmh": 50} -> regra dos 2 s; {"traseira": False}
                           #      para vídeos de câmara FRONTAL (lados não espelhados);
                           #      {"dist_seguranca": 15}; LINHAS_PARAMS {"altura": 1.0}
                           #      (altura da câmara, em m) acerta a escala das distâncias
ESTACIONAMENTO = False     # guias de distância no chão (também: tecla G no player)
MOSTRAR = os.environ.get("VA_MOSTRAR", "1") != "0"     # abrir o player no fim
GRAFICOS = os.environ.get("VA_GRAFICOS", "1") != "0"   # mostrar gráficos
GRAVAR_VIDEO = os.environ.get("VA_GRAVAR_VIDEO", "0") != "0"  # gravar vídeo anotado (mp4)
FICH_VIDEO = "dados/resultado_percecao.mp4"
MAX_FRAMES = int(os.environ.get("VA_MAX_FRAMES", "0")) # 0 = todas (ex. 50 p/ testar)
FICH_GT = ""               # ground truth .json (opcional) -> avaliação de métricas
# ==========================================================================


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # corre a partir do Projeto/
    import cv2
    from percecao import fontes, detecao, aproximacao, exportar, player, desenho
    from percecao import sinais as msinais
    from percecao.linhas import DetetorLinhas
    from percecao.seguranca import AnaliseSeguranca

    fonte = fontes.Fonte(FONTE)
    print(f"Fonte: {fonte.tipo} | {fonte.largura}x{fonte.altura} | "
          f"{fonte.n} frames | {fonte.fps:.1f} fps")

    det = detecao.Detetor(MODELO, conf=CONF, imgsz=IMGSZ, usar_track=USAR_TRACK)
    det_linhas = DetetorLinhas(LINHAS_PARAMS) if LINHAS else None

    rec_sinais = None
    if os.path.isfile(SINAIS_DET):
        from percecao.sinais_treinados import ReconhecedorSinais
        rec_sinais = ReconhecedorSinais(SINAIS_DET, SINAIS_CLS, imgsz=IMGSZ)
        print("Sinais: modelos TREINADOS (deteção GTSDB + 43 tipos GTSRB)")
    elif SINAIS_CLASSICO:
        print("Sinais: YOLO + detetor clássico (corre treinar_sinais.py para 43 tipos)")
    aprox = aproximacao.Aproximacao(limiar_risco=LIMIAR_RISCO) if ANALISAR_APROX else None
    seguranca = AnaliseSeguranca(dict(SEGURANCA_PARAMS, ttc_risco=LIMIAR_RISCO),
                                 LINHAS_PARAMS) if SEGURANCA else None
    memoria_sinais = desenho.MemoriaSinais()

    vw = None
    resultados, stats = [], []
    t0 = time.time()
    for idx, frame in fonte.iterar():
        t = idx / (fonte.fps or 15.0)
        res = det.detetar(frame)
        if rec_sinais is not None:        # modelos treinados (43 tipos)
            res["sinais"] = msinais.combinar(res["sinais"], rec_sinais.detetar(frame))
        elif SINAIS_CLASSICO:             # alternativa clássica (cor + forma)
            res["sinais"] = msinais.combinar(res["sinais"], msinais.detetar(frame))
        if det_linhas is not None:
            # as caixas de veículos/peões são ignoradas (as arestas deles não são linhas)
            res["linhas"] = det_linhas.detetar(
                frame, excluir=res["veic"]["bb"] + res["duas_rodas"]["bb"] + res["peoes"]["bb"])

        ttc_min = float("inf")
        if aprox is not None:
            for g in ("veic", "duas_rodas"):
                res[g]["ttc"] = aprox.atualizar(res[g], t, g)
            ttc_min = aprox.ttc_minimo(res["veic"]["ttc"] + res["duas_rodas"]["ttc"])
        seg = seguranca.analisar(frame.shape, res, t) if seguranca is not None else {}
        res["sinais_vistos"] = memoria_sinais.atualizar(frame, res["sinais"], t)
        seguidor = seg.get("seguidor") or {}
        mud = seg.get("mudanca", {})

        resultados.append(res)
        lin = res.get("linhas") or {}
        stats.append(dict(frame=idx,
                          t=t,
                          nlinhas=lin.get("n", 0),
                          tipo_esq=(lin.get("esq") or {}).get("tipo"),     # aula 5, ex. 4a
                          tipo_dir=(lin.get("dir") or {}).get("tipo"),
                          marcacoes="+".join(m["tipo"] for m in lin.get("marcacoes", [])),
                          nveic=len(res["veic"]["bb"]),
                          nduas=len(res["duas_rodas"]["bb"]),
                          npeoes=len(res["peoes"]["bb"]),
                          nsinais=len(res["sinais"]["bb"]),
                          ttc_min=ttc_min,
                          risco=bool(ttc_min < LIMIAR_RISCO),
                          dist_atras=seguidor.get("d"),
                          estado_dist=seg.get("estado_dist"),
                          mudar_esq=mud.get("esquerda", (None,))[0],
                          mudar_dir=mud.get("direita", (None,))[0],
                          livre_atras=seg.get("livre_atras"),
                          alertas=" | ".join(a for _, a in seg.get("alertas", []))))
        if GRAVAR_VIDEO:
            anot = desenho.desenhar(frame, res, ttc_risco=LIMIAR_RISCO,
                                    estacionamento=ESTACIONAMENTO)
            if vw is None:
                os.makedirs(os.path.dirname(FICH_VIDEO), exist_ok=True)
                vw = cv2.VideoWriter(FICH_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fonte.fps or 15.0, (anot.shape[1], anot.shape[0]))
            vw.write(anot)
        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1} frames", flush=True)
        if MAX_FRAMES and idx + 1 >= MAX_FRAMES:
            break
    if vw is not None:
        vw.release()
        print(f"Vídeo anotado gravado em {FICH_VIDEO}")
    dt = time.time() - t0
    nfp = len(resultados)
    print(f"\nProcessadas {nfp} frames em {dt:.1f}s ({nfp / max(dt, 1e-9):.1f} fps).")

    exportar.exportar_csv(resultados, "deteccoes.csv")
    if det_linhas is not None:
        exportar.exportar_linhas_csv(stats, "linhas.csv")     # aula 5, ex. 4a
    exportar.exportar_stats_csv(stats, "seguranca.csv")       # avisos/distâncias por frame
    _resumo(stats)

    if FICH_GT and os.path.isfile(FICH_GT):
        from percecao import avaliacao
        with open(FICH_GT, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print("\n== Avaliação vs ground truth ==")
        avaliacao.avaliar(resultados, gt)

    # Player PRIMEIRO (o vídeo é o principal); os gráficos abrem só depois de o fechar.
    if MOSTRAR:
        player.reproduzir(fonte, resultados, stats, fps=fonte.fps, ttc_risco=LIMIAR_RISCO,
                          estacionamento=ESTACIONAMENTO)
    _graficos(stats, GRAFICOS)
    fonte.fechar()


def _resumo(stats):
    if not stats:
        print("Sem frames processadas."); return
    n = len(stats)
    import statistics as st
    print(f"\n===== Resumo ({n} frames) =====")
    nl = [s.get("nlinhas", 0) for s in stats]
    print(f"Linhas: 2 -> {100 * nl.count(2) / n:.0f}%  |  1 -> {100 * nl.count(1) / n:.0f}%"
          f"  |  0 -> {100 * nl.count(0) / n:.0f}%  (das frames)")
    for lado, nome in (("tipo_esq", "esquerda"), ("tipo_dir", "direita ")):
        tipos = [s.get(lado) for s in stats if s.get(lado)]
        if tipos:
            print(f"  linha {nome}: vista em {100 * len(tipos) / n:.0f}% | "
                  f"contínua {100 * tipos.count('continua') / len(tipos):.0f}% · "
                  f"tracejada {100 * tipos.count('tracejada') / len(tipos):.0f}%")
    marc = [s.get("marcacoes") for s in stats if s.get("marcacoes")]
    if marc:
        print(f"Marcações no pavimento: passadeira em {sum('passadeira' in m for m in marc)} frames,"
              f" linha de paragem em {sum('linha_paragem' in m for m in marc)} frames")
    print(f"Veículos/frame (média): {st.mean(s['nveic'] for s in stats):.2f}")
    print(f"2 rodas/frame (média) : {st.mean(s['nduas'] for s in stats):.2f}")
    print(f"Peões/frame (média)   : {st.mean(s['npeoes'] for s in stats):.2f}")
    print(f"Sinais/frame (média)  : {st.mean(s['nsinais'] for s in stats):.2f}")
    est = [s.get("estado_dist") for s in stats if s.get("estado_dist")]
    if est:
        print("Veículo atrás (na nossa faixa): " + " · ".join(
            f"{k} {100 * est.count(k) / n:.0f}%" for k in ("ok", "atencao", "perigo")) +
            f"  | livre {100 * est.count('livre') / n:.0f}%")
        for lado in ("esq", "dir"):
            m = [s.get(f"mudar_{lado}") for s in stats]
            print(f"  mudar p/ {'esquerda' if lado == 'esq' else 'direita '}: " + " · ".join(
                f"{k} {100 * m.count(k) / n:.0f}%" for k in ("livre", "proibido", "ocupado")))
        nal = sum(1 for s in stats if s.get("alertas"))
        print(f"Frames com avisos: {nal} ({100 * nal / n:.0f}%)")
    riscos = sum(1 for s in stats if s["risco"])
    ttcs = [s["ttc_min"] for s in stats if s["ttc_min"] != float("inf")]
    if ttcs:
        print(f"Aproximação: TTC mínimo = {min(ttcs):.1f}s | frames em risco = "
              f"{riscos} ({100 * riscos / n:.0f}%)")


def _graficos(stats, mostrar):
    if not stats:
        return
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")         # sem janela (só grava PNG)
    import matplotlib.pyplot as plt
    os.makedirs("dados", exist_ok=True)
    x = [s["frame"] for s in stats]

    f1 = plt.figure("Deteções por frame")
    plt.plot(x, [s.get("nlinhas", 0) for s in stats], label="Linhas", alpha=0.7)
    plt.plot(x, [s["nveic"] for s in stats], label="Veículos")
    plt.plot(x, [s["nduas"] for s in stats], label="2 rodas")
    plt.plot(x, [s["npeoes"] for s in stats], label="Peões")
    plt.plot(x, [s["nsinais"] for s in stats], label="Sinais")
    plt.xlabel("Frame"); plt.ylabel("Nº de deteções"); plt.grid(True); plt.legend()
    plt.title("Deteções por frame")
    f1.savefig("dados/graf_deteccoes.png", dpi=120, bbox_inches="tight")

    y = [s["ttc_min"] if s["ttc_min"] != float("inf") else float("nan") for s in stats]
    if any(v == v for v in y):        # existe algum não-NaN
        f2 = plt.figure("Aproximação (TTC)")
        plt.plot(x, y, "b-")
        plt.axhline(LIMIAR_RISCO, color="r", linestyle="--", label="limiar de risco")
        plt.xlabel("Frame"); plt.ylabel("TTC mínimo (s)"); plt.grid(True); plt.legend()
        plt.title("Tempo para colisão do veículo mais próximo por trás")
        f2.savefig("dados/graf_ttc.png", dpi=120, bbox_inches="tight")

    d = [s["dist_atras"] if s.get("dist_atras") is not None else float("nan") for s in stats]
    if any(v == v for v in d):
        f3 = plt.figure("Distância ao veículo de trás")
        plt.plot(x, d, "k-", lw=1, label="distância (monocular, aprox.)")
        cores = {"ok": "tab:green", "atencao": "tab:orange", "perigo": "tab:red"}
        for k, c in cores.items():
            xs = [xi for xi, s in zip(x, stats) if s.get("estado_dist") == k]
            ys = [di for di, s in zip(d, stats) if s.get("estado_dist") == k]
            plt.scatter(xs, ys, s=8, c=c, label=k)
        plt.xlabel("Frame"); plt.ylabel("Distância (m)"); plt.grid(True); plt.legend()
        plt.title("Distância de segurança ao veículo que segue atrás (nossa faixa)")
        f3.savefig("dados/graf_distancia.png", dpi=120, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
