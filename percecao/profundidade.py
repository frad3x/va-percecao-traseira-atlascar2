"""
profundidade.py — profundidade (RGB-D) -> nuvem de pontos.

- nuvem_de_profundidade(D, K): reconstrói XYZ de uma imagem de profundidade + K.
- bag_para_plys(fonte, saida_dir): lê a profundidade de um rosbag/MCAP e grava
  uma nuvem .ply por frame (para o percecao3d.analisar).

D em metros (float) ou milímetros (uint16 -> /1000). K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
"""
import glob
import os

import numpy as np


def nuvem_de_profundidade(D, K):
    D = np.asarray(D)
    Z = D.astype(np.float32) / 1000.0 if np.issubdtype(D.dtype, np.integer) else D.astype(np.float32)
    K = np.asarray(K, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    val = np.isfinite(Z) & (Z > 0)
    return np.stack([X[val], Y[val], Z[val]], axis=1).astype(np.float32)


# ---------------- leitura de um rosbag/MCAP ----------------
def _encontrar_mcap(fonte):
    if os.path.isfile(fonte) and fonte.lower().endswith(".mcap"):
        return fonte
    if os.path.isdir(fonte):
        achados = sorted(glob.glob(os.path.join(fonte, "*.mcap")))
        if achados:
            return achados[0]
    return None


def _topicos(mcap_path):
    from mcap.reader import make_reader
    prof = info = None
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
        schemas = {sid: s.name for sid, s in summary.schemas.items()}
        for c in summary.channels.values():
            nome = schemas.get(c.schema_id, "")
            t = c.topic.lower()
            if prof is None and nome.endswith("msg/Image") and "depth" in t:
                prof = c.topic
            if info is None and nome.endswith("msg/CameraInfo") and "depth" in t:
                info = c.topic
        if prof is None:
            for c in summary.channels.values():
                if schemas.get(c.schema_id, "").endswith("msg/Image"):
                    prof = c.topic; break
        if info is None:
            for c in summary.channels.values():
                if schemas.get(c.schema_id, "").endswith("msg/CameraInfo"):
                    info = c.topic; break
    return prof, info


def _intrinsecos(mcap_path, topico_info):
    if topico_info is None:
        return None
    from mcap_ros2.reader import read_ros2_messages
    for m in read_ros2_messages(mcap_path, topics=[topico_info]):
        k = getattr(m.ros_msg, "k", None)
        if k is not None and len(k) == 9 and k[0] > 0:
            return np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], float)
    return None


def _depth_msg_xyz(msg, K):
    h, w = int(msg.height), int(msg.width)
    enc = getattr(msg, "encoding", "16UC1")
    buf = bytes(msg.data)
    if enc in ("16UC1", "mono16"):
        z = np.frombuffer(buf, dtype=np.uint16).astype(np.float32) / 1000.0
    elif enc == "32FC1":
        z = np.frombuffer(buf, dtype=np.float32).copy()
    else:
        raise ValueError(f"Encoding de profundidade não suportado: {enc}")
    return nuvem_de_profundidade(z[:h * w].reshape(h, w), K)


def bag_para_plys(fonte, saida_dir, passo=5, maximo=0):
    """Lê a profundidade de um bag e grava uma nuvem .ply por frame. Devolve nº gravadas."""
    import open3d as o3d
    mcap_path = _encontrar_mcap(fonte)
    assert mcap_path, f"Sem .mcap em {fonte}"
    prof, info = _topicos(mcap_path)
    assert prof, "Sem tópico de profundidade no bag (gravaste com o driver Orbbec?)."
    K = _intrinsecos(mcap_path, info)
    assert K is not None, "Sem CameraInfo/intrínsecos da profundidade no bag."
    os.makedirs(saida_dir, exist_ok=True)
    print(f"[profundidade] depth={prof} info={info}")

    from mcap_ros2.reader import read_ros2_messages
    n = grav = 0
    for m in read_ros2_messages(mcap_path, topics=[prof]):
        if n % max(passo, 1) == 0:
            pts = _depth_msg_xyz(m.ros_msg, K)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            o3d.io.write_point_cloud(os.path.join(saida_dir, f"nuvem_{n:05d}.ply"), pc)
            grav += 1
            if maximo and grav >= maximo:
                break
        n += 1
    print(f"[profundidade] OK: {grav} nuvens .ply em {saida_dir}")
    return grav


# --- CLI:  python -m percecao.profundidade --fonte <bag> --saida_dir dados/nuvens ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Profundidade de um rosbag/MCAP -> nuvens .ply")
    ap.add_argument("--fonte", required=True, help="pasta do rosbag2 ou ficheiro .mcap")
    ap.add_argument("--saida_dir", required=True, help="pasta onde gravar os .ply")
    ap.add_argument("--passo", type=int, default=5, help="processa 1 em cada N frames")
    ap.add_argument("--max", type=int, default=0, help="nº máx. de nuvens (0 = todas)")
    a = ap.parse_args()
    bag_para_plys(a.fonte, a.saida_dir, passo=a.passo, maximo=a.max)
