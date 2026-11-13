"""
app.py — a APLICAÇÃO de processamento off-line (vídeo / pasta de imagens / rosbag).

Percorre a fonte uma vez com o Pipeline (o mesmo núcleo que o nó ROS 2 usa),
exporta CSV, mostra o resumo, abre o player e os gráficos.

Formas de a correr:
  - main.py (na raiz do projeto): edita o bloco CONFIG e dá Run;
  - linha de comandos:  python -m percecao <fonte> [opções]   (ver --help)
  - depois de 'pip install .':  va-percecao <fonte> [opções]
"""
import datetime
import json
import os
import time

from .pipeline import Config, Pipeline

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pasta do projeto
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")


def nova_pasta_resultados(fonte_path):
    """resultados/<nome da gravação>_<data_hora>/ — uma pasta por análise."""
    nome = os.path.splitext(os.path.basename(str(fonte_path).rstrip("/\\")))[0] or "analise"
    pasta = os.path.join(PASTA_RESULTADOS, f"{nome}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(pasta, exist_ok=True)
    return pasta


def processar(fonte, cfg=None, progresso=None, parar=None, gravar_video=False,
              fich_video="video_anotado.mp4", max_frames=0, estacionamento=False,
              log=print):
    """Percorre a fonte (já aberta) com o Pipeline.
    progresso(k, n, res, frame) é chamado a cada imagem; parar() -> True interrompe."""
    import cv2
    from . import desenho
    cfg = cfg or Config()
    pipe = Pipeline(cfg, log=log)
    vw = None
    resultados, stats = [], []
    t0 = time.time()
    n_total = min(fonte.n, max_frames) if max_frames and fonte.n else (max_frames or fonte.n)
    for idx, frame in fonte.iterar():
        res = pipe.processar(frame, idx / (fonte.fps or 15.0))
        resultados.append(res)
        stats.append(pipe.estatisticas(res, idx))
        if gravar_video:
            anot = desenho.desenhar(frame, res, ttc_risco=cfg.limiar_risco,
                                    estacionamento=estacionamento)
            if vw is None:
                os.makedirs(os.path.dirname(fich_video) or ".", exist_ok=True)
                vw = cv2.VideoWriter(fich_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fonte.fps or 15.0, (anot.shape[1], anot.shape[0]))
            vw.write(anot)
        if progresso is not None:
            progresso(idx, n_total, res, frame)
        elif (idx + 1) % 100 == 0:
            log(f"... {idx + 1} frames")
        if (max_frames and idx + 1 >= max_frames) or (parar is not None and parar()):
            break
    if vw is not None:
        vw.release()
        log(f"Vídeo anotado gravado em {fich_video}")
    dt = time.time() - t0
    log(f"Processadas {len(resultados)} frames em {dt:.1f}s ({len(resultados) / max(dt, 1e-9):.1f} fps).")
    return resultados, stats


def exportar(resultados, stats, cfg, pasta="."):
    """Escreve os 3 CSV na pasta dada."""
    from . import exportar as ex
    os.makedirs(pasta, exist_ok=True)
    ex.exportar_csv(resultados, os.path.join(pasta, "deteccoes.csv"))
    if cfg.linhas:
        ex.exportar_linhas_csv(stats, os.path.join(pasta, "linhas.csv"))     # aula 5, ex. 4a
    ex.exportar_stats_csv(stats, os.path.join(pasta, "seguranca.csv"))       # avisos por frame


def executar(fonte_path, cfg=None, mostrar=True, graficos=True, gravar_video=False,
             fich_video=None, max_frames=0, fich_gt="", estacionamento=False):
    """Tudo o que esta análise produz (CSV, gráficos, vídeo) vai para
    resultados/<gravação>_<data>/ — a pasta do projeto fica limpa."""
    from . import fontes, player
    cfg = cfg or Config()
    fonte = fontes.Fonte(fonte_path)
    print(f"Fonte: {fonte.tipo} | {fonte.largura}x{fonte.altura} | "
          f"{fonte.n} frames | {fonte.fps:.1f} fps")
    pasta = nova_pasta_resultados(fonte_path)
    fich_video = fich_video or os.path.join(pasta, "video_anotado.mp4")
    resultados, stats = processar(fonte, cfg, gravar_video=gravar_video, fich_video=fich_video,
                                  max_frames=max_frames, estacionamento=estacionamento)
    exportar(resultados, stats, cfg, pasta)
    print(f"Resultados em {pasta}")
    _resumo(stats)

    if fich_gt and os.path.isfile(fich_gt):
        from . import avaliacao
        with open(fich_gt, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print("\n== Avaliação vs ground truth ==")
        avaliacao.avaliar(resultados, gt)

    # Player PRIMEIRO (o vídeo é o principal); os gráficos abrem só depois de o fechar.
    if mostrar:
        player.reproduzir(fonte, resultados, stats, fps=fonte.fps, ttc_risco=cfg.limiar_risco,
                          estacionamento=estacionamento)
    _graficos(stats, graficos, cfg.limiar_risco, pasta=pasta)
    fonte.fechar()
    return resultados, stats


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


def _graficos(stats, mostrar, LIMIAR_RISCO=4.0, pasta=PASTA_RESULTADOS, bloquear=True):
    """Gráficos (PNG na pasta dada). bloquear=False: dentro de uma app Qt já a correr."""
    if not stats:
        return
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")         # sem janela (só grava PNG)
    import matplotlib.pyplot as plt
    os.makedirs(pasta, exist_ok=True)
    x = [s["frame"] for s in stats]

    f1 = plt.figure("Deteções por frame")
    plt.plot(x, [s.get("nlinhas", 0) for s in stats], label="Linhas", alpha=0.7)
    plt.plot(x, [s["nveic"] for s in stats], label="Veículos")
    plt.plot(x, [s["nduas"] for s in stats], label="2 rodas")
    plt.plot(x, [s["npeoes"] for s in stats], label="Peões")
    plt.plot(x, [s["nsinais"] for s in stats], label="Sinais")
    plt.xlabel("Frame"); plt.ylabel("Nº de deteções"); plt.grid(True); plt.legend()
    plt.title("Deteções por frame")
    f1.savefig(os.path.join(pasta, "graf_deteccoes.png"), dpi=120, bbox_inches="tight")

    y = [s["ttc_min"] if s["ttc_min"] != float("inf") else float("nan") for s in stats]
    if any(v == v for v in y):        # existe algum não-NaN
        f2 = plt.figure("Aproximação (TTC)")
        plt.plot(x, y, "b-")
        plt.axhline(LIMIAR_RISCO, color="r", linestyle="--", label="limiar de risco")
        plt.xlabel("Frame"); plt.ylabel("TTC mínimo (s)"); plt.grid(True); plt.legend()
        plt.title("Tempo para colisão do veículo mais próximo por trás")
        f2.savefig(os.path.join(pasta, "graf_ttc.png"), dpi=120, bbox_inches="tight")

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
        f3.savefig(os.path.join(pasta, "graf_distancia.png"), dpi=120, bbox_inches="tight")

    if mostrar:
        plt.show(block=bloquear)
    else:
        plt.close("all")
