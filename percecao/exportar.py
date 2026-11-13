"""exportar.py — escreve todas as deteções num CSV (uma linha por caixa)."""


def exportar_csv(resultados, caminho):
    grupos = ("veic", "peoes", "sinais")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("frame,grupo,classe,x,y,w,h,conf\n")
        for k, res in enumerate(resultados):
            for g in grupos:
                gr = res.get(g, {})
                bbs = gr.get("bb", [])
                for j, bb in enumerate(bbs):
                    tipo = gr.get("tipo", [])
                    cl = tipo[j] if j < len(tipo) else ""
                    sc = gr["sc"][j]
                    f.write(f"{k},{g},{cl},{bb[0]:.1f},{bb[1]:.1f},"
                            f"{bb[2]:.1f},{bb[3]:.1f},{sc:.3f}\n")
    print(f"Deteções exportadas para {caminho}")


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
