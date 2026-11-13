"""exportar.py — escreve todas as deteções num CSV (uma linha por caixa)."""


def exportar_csv(resultados, caminho):
    grupos = ("veic", "duas_rodas", "peoes", "sinais")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("frame,grupo,classe,id,x,y,w,h,conf,dist_m,faixa,ttc_s\n")
        for k, res in enumerate(resultados):
            for g in grupos:
                gr = res.get(g, {})
                bbs = gr.get("bb", [])
                for j, bb in enumerate(bbs):
                    tipo = gr.get("tipo", [])
                    cl = tipo[j] if j < len(tipo) else ""
                    sc = gr["sc"][j]
                    oid = _campo(gr, "id", j, "")
                    d = _campo(gr, "dist", j)
                    faixa = _campo(gr, "faixa", j, "") or ""
                    ttc = _campo(gr, "ttc", j)
                    f.write(f"{k},{g},{cl},{oid},{bb[0]:.1f},{bb[1]:.1f},"
                            f"{bb[2]:.1f},{bb[3]:.1f},{sc:.3f},"
                            f"{'' if d is None else f'{d:.1f}'},{faixa},"
                            f"{'' if ttc is None or ttc == float('inf') else f'{ttc:.2f}'}\n")
    print(f"Deteções exportadas para {caminho}")


def _campo(gr, chave, j, padrao=None):
    lst = gr.get(chave)
    return lst[j] if lst is not None and j < len(lst) else padrao


def exportar_stats_csv(stats, caminho):
    """Uma linha por frame com o resumo (contagens, distância, avisos...)."""
    if not stats:
        return
    cols = list(stats[0].keys())
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for s in stats:
            vals = []
            for c in cols:
                v = s.get(c)
                if v is None or v == float("inf"):
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.3f}"
                vals.append(str(v).replace(",", ";"))
            f.write(",".join(vals) + "\n")
    print(f"Resumo por frame (avisos, distâncias) exportado para {caminho}")


def exportar_linhas_csv(stats, caminho):
    """Uma linha por frame, com a estrutura sugerida na aula 5 (ex. 4a):
    frameID, frameTime, leftLine, leftLineType, rightLine, rightLineType."""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("frameID,frameTime,leftLine,leftLineType,rightLine,rightLineType\n")
        for s in stats:
            te, td = s.get("tipo_esq"), s.get("tipo_dir")
            f.write(f"{s['frame']},{s.get('t', 0):.3f},{int(bool(te))},{te or ''},"
                    f"{int(bool(td))},{td or ''}\n")
    print(f"Linhas por frame exportadas para {caminho}")
