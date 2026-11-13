"""
percecao3d.py — perceção 3D (RGB-D) para apoio ao estacionamento, com Open3D.

Porte de percecao3D.m. Pipeline das teses do ATLASCAR:
  1) VOXEL GRID            (voxel_down_sample)
  2) PLANO DO CHÃO (RANSAC)(segment_plane) -> separa chão de obstáculos
  3) SEGMENTAÇÃO           (cluster_dbscan) -> obstáculos
  4) LUGAR DE ESTACIONAMENTO -> verifica se uma ROI ao lado do carro está LIVRE

Uso:
  from percecao import percecao3d
  s = percecao3d.analisar('demo')                 # cena sintética
  s = percecao3d.analisar('nuvem.ply')            # ficheiro
  s = percecao3d.analisar({'depth': D, 'K': K})   # profundidade + intrínsecos
  s = percecao3d.analisar(pontos_Nx3)             # array numpy
"""
import numpy as np
import open3d as o3d

CFG = dict(voxel=0.10, dist_chao=0.08, max_dist=20.0,
           min_pts_cluster=30, dist_cluster=0.5,
           roi_lugar=None, max_pts_livre=40, mostrar=True)


def analisar(entrada, cfg=None):
    c = dict(CFG)
    if cfg:
        c.update(cfg)
    eh_demo = isinstance(entrada, str) and entrada.lower() == "demo"
    if eh_demo and c["roi_lugar"] is None:
        c["roi_lugar"] = [0.5, 3.5, 0.0, 1.6, 5.2, 6.8]   # ROI sobre o vazio da demo

    pc = _obter_nuvem(entrada)

    # 1) voxel + distância
    pc = pc.voxel_down_sample(c["voxel"])
    pts = np.asarray(pc.points)
    if len(pts):
        d = np.linalg.norm(pts, axis=1)
        pc = pc.select_by_index(np.where(d <= c["max_dist"])[0])

    # 2) plano do chão (RANSAC)
    plano, idx_chao = pc.segment_plane(c["dist_chao"], ransac_n=3, num_iterations=1000)
    obst = pc.select_by_index(idx_chao, invert=True)
    chao = pc.select_by_index(idx_chao)

    # 3) clustering de obstáculos
    obstaculos = []
    L = np.asarray(obst.points)
    if len(L):
        lab = np.array(obst.cluster_dbscan(eps=c["dist_cluster"],
                                           min_points=c["min_pts_cluster"]))
        for cl in range(lab.max() + 1 if lab.size else 0):
            P = L[lab == cl]
            if len(P) < c["min_pts_cluster"]:
                continue
            obstaculos.append(dict(centro=P.mean(0).tolist(),
                                   bbox=np.concatenate([P.min(0), P.max(0)]).tolist(),
                                   n=len(P)))

    # 4) lugar de estacionamento (ROI livre?)
    roi = c["roi_lugar"] or [0.0, 4.5, -3.5, -1.2, -0.2, 2.0]
    n_roi = 0
    if len(L):
        dentro = ((L[:, 0] >= roi[0]) & (L[:, 0] <= roi[1]) &
                  (L[:, 1] >= roi[2]) & (L[:, 1] <= roi[3]) &
                  (L[:, 2] >= roi[4]) & (L[:, 2] <= roi[5]))
        n_roi = int(dentro.sum())
    livre = n_roi <= c["max_pts_livre"]

    print(f"percecao3d: {len(obstaculos)} obstaculos | ROI tem {n_roi} pontos "
          f"-> lugar {'LIVRE' if livre else 'OCUPADO'}")

    if c["mostrar"]:
        _desenhar(chao, obst, obstaculos, roi, livre)

    return dict(plano=plano, nuvem_obst=obst, obstaculos=obstaculos,
                lugar_livre=livre, roi=roi, n_pontos_roi=n_roi)


# ------------------------- entrada -------------------------
def _obter_nuvem(entrada):
    if isinstance(entrada, o3d.geometry.PointCloud):
        return entrada
    if isinstance(entrada, str):
        if entrada.lower() == "demo":
            return _nuvem_demo()
        return o3d.io.read_point_cloud(entrada)
    if isinstance(entrada, dict) and "depth" in entrada and "K" in entrada:
        from .profundidade import nuvem_de_profundidade
        return _pc(nuvem_de_profundidade(entrada["depth"], entrada["K"]))
    if isinstance(entrada, np.ndarray):
        return _pc(entrada)
    raise ValueError("Entrada 3D não suportada (usa PointCloud, ficheiro, dict depth+K, array ou 'demo').")


def _pc(arr):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(arr, dtype=np.float64))
    return pc


def _nuvem_demo():
    rng = np.random.default_rng(0)
    gx, gz = np.meshgrid(np.arange(-4, 6, 0.05), np.arange(0, 12, 0.05))
    chao = np.stack([gx.ravel(), 0.01 * rng.standard_normal(gx.size), gz.ravel()], 1)
    car1 = _caixa([2, 0, 3], [1.6, 1.4, 4.0], 1500, rng)
    car2 = _caixa([2, 0, 9], [1.6, 1.4, 4.0], 1500, rng)
    return _pc(np.vstack([chao, car1, car2]))


def _caixa(centro, dim, n, rng):
    P = (rng.random((n, 3)) - 0.5) * np.array(dim) + np.array(centro)
    P[:, 1] += dim[1] / 2
    return P


# ------------------------- visualização -------------------------
def _desenhar(chao, obst, obstaculos, roi, livre):
    chao.paint_uniform_color([0.5, 0.5, 0.5])
    geoms = [chao]
    if len(np.asarray(obst.points)):
        obst.paint_uniform_color([1.0, 0.3, 0.3])
        geoms.append(obst)
    for o in obstaculos:
        geoms.append(_caixa_linhas(o["bbox"], [1, 1, 0]))
    geoms.append(_caixa_linhas(roi, [0, 1, 0] if livre else [1, 0, 0]))
    o3d.visualization.draw_geometries(geoms, window_name="Percecao 3D - estacionamento")


def _caixa_linhas(bbox, cor):
    mn, mx = bbox[:3], bbox[3:]
    V = [[mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
         [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]]]
    E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(V), o3d.utility.Vector2iVector(E))
    ls.colors = o3d.utility.Vector3dVector([cor] * len(E))
    return ls


if __name__ == "__main__":
    analisar("demo")
