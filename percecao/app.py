"""
app.py — a APLICAÇÃO de processamento off-line (vídeo / pasta de imagens / rosbag).

Percorre a fonte uma vez com o Pipeline (o mesmo núcleo que o nó ROS 2 usa),
exporta CSV, mostra o resumo, abre o player e os gráficos.

Formas de a correr:
  - main.py (na raiz do projeto): edita o bloco CONFIG e dá Run;
  - linha de comandos:  python -m percecao <fonte> [opções]   (ver --help)
  - depois de 'pip install .':  va-percecao <fonte> [opções]
"""
import json
import os
import time

from .pipeline import Config, Pipeline


def executar(fonte_path, cfg=None, mostrar=True, graficos=True, gravar_video=False,
             fich_video="dados/resultado_percecao.mp4", max_frames=0, fich_gt="",
             estacionamento=False):
    cfg = cfg or Config()
    import cv2
    from . import fontes, exportar, player, desenho

    fonte = fontes.Fonte(fonte_path)
    print(f"Fonte: {fonte.tipo} | {fonte.largura}x{fonte.altura} | "
          f"{fonte.n} frames | {fonte.fps:.1f} fps")

    pipe = Pipeline(cfg)
    LIMIAR_RISCO = cfg.limiar_risco

    vw = None
    resultados, stats = [], []
    t0 = time.time()
    for idx, frame in fonte.iterar():
        res = pipe.processar(frame, idx / (fonte.fps or 15.0))
        resultados.append(res)
        stats.append(pipe.estatisticas(res, idx))
        if gravar_video:
            anot = desenho.desenhar(frame, res, ttc_risco=LIMIAR_RISCO,
                                    estacionamento=estacionamento)
            if vw is None:
                os.makedirs(os.path.dirname(fich_video), exist_ok=True)
                vw = cv2.VideoWriter(fich_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fonte.fps or 15.0, (anot.shape[1], anot.shape[0]))
            vw.write(anot)
        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1} frames", flush=True)
        if max_frames and idx + 1 >= max_frames:
            break
    if vw is not None:
        vw.release()
        print(f"Vídeo anotado gravado em {fich_video}")
    dt = time.time() - t0
    nfp = len(resultados)
    print(f"\nProcessadas {nfp} frames em {dt:.1f}s ({nfp / max(dt, 1e-9):.1f} fps).")

    exportar.exportar_csv(resultados, "deteccoes.csv")
    if cfg.linhas:
        exportar.exportar_linhas_csv(stats, "linhas.csv")     # aula 5, ex. 4a
    exportar.exportar_stats_csv(stats, "seguranca.csv")       # avisos/distâncias por frame
    _resumo(stats)

    if fich_gt and os.path.isfile(fich_gt):
        from . import avaliacao
        with open(fich_gt, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print("\n== Avaliação vs ground truth ==")
        avaliacao.avaliar(resultados, gt)

    # Player PRIMEIRO (o vídeo é o principal); os gráficos abrem só depois de o fechar.
    if mostrar:
        player.reproduzir(fonte, resultados, stats, fps=fonte.fps, ttc_risco=LIMIAR_RISCO,
                          estacionamento=estacionamento)
    _graficos(stats, graficos, LIMIAR_RISCO)
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


def _graficos(stats, mostrar, LIMIAR_RISCO=4.0):
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
