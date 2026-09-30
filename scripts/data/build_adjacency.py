"""Genera la matriz de adyacencia 111x111 del esqueleto (7 pose + 64 cara + 20+20 manos).

Reemplaza el artefacto `adjacency_matrix.npy` que estaba hardcodeado a una ruta de otra
máquina en setup_train.build_model. Reutiliza las definiciones de aristas de
`KeypointProcessingGraph` (BODY_EDGES + HAND_TEMPLATE con offsets 70/90), evitando el assert
roto de su build_skeleton_graph. La adyacencia es estructural (no depende de datos reales).

Uso:
    PYTHONPATH=. python scripts/data/build_adjacency.py [--out data/processed/adjacency_matrix.npy]
"""
import argparse
from pathlib import Path

import networkx as nx
import numpy as np

from src.mslm.dataloader.graph_keypoints import KeypointProcessingGraph

N_NODES = 111  # 71 (7 pose + 64 cara) + 20 mano izq + 20 mano der


def build_adjacency() -> np.ndarray:
    kpg = KeypointProcessingGraph()
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))

    # Cuerpo/cara
    G.add_edges_from(kpg.BODY_EDGES)

    # Mano izquierda (offset 70) y derecha (offset 90), conectadas a la muñeca del cuerpo.
    left_off, right_off = 70, 90
    G.add_edges_from([(6, left_off)] + [(u + left_off, v + left_off) for u, v in kpg.HAND_TEMPLATE])
    G.add_edges_from([(3, right_off)] + [(u + right_off, v + right_off) for u, v in kpg.HAND_TEMPLATE])

    A = nx.adjacency_matrix(G, nodelist=range(N_NODES)).todense()
    return np.asarray(A, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed/adjacency_matrix.npy"),
                        help="Ruta de salida (relativa al padre del repo si empieza con 'data/').")
    args = parser.parse_args()

    # data/ vive en el padre del repo (igual que paths.py)
    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    out.parent.mkdir(parents=True, exist_ok=True)

    A = build_adjacency()
    np.save(out, A)
    print(f"Adyacencia {A.shape}, aristas={int(A.sum() // 2)}, guardada en: {out}")
