import networkx as nx
from typing import Optional, Tuple
import numpy as np

class KeypointProcessingGraph():
    def __init__(self):
        self.BODY_EDGES = [
            (0, 1), (0, 4), (1, 4),
            (1, 2), (2, 3), (4, 5), (5, 6),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), 
            (19, 20), (20, 21), (21, 22), (22, 23),                       # ceja derecha
            (24, 25), (25, 26), (26, 27), (27, 28),                       # ceja izquierda
            (38, 39), (39, 40), (40, 41), (41, 42), (42, 43),             # ojo derecho
            (44, 45), (45, 46), (46, 47), (47, 48), (48, 49),             # ojo izquierdo
            (50, 51), (51, 52), (52, 53), (53, 54), (54, 55),             # labio superior
            (56, 57), (57, 58), (58, 59), (59, 60), (61, 62), (62, 63),   # labio inferior
            (64, 65), (65, 66), (66, 67), (67, 68), (68, 69),             # se parador labio
        ]
        
        self.HAND_TEMPLATE = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]

        self.body_kpts = 71
        self.hand_kpts = 20

    def build_skeleton_graph(self, kpts, return_adjacency=False) -> Tuple[nx.Graph, Optional[np.ndarray]]:
        """
        kpts: Tensor of shape [N, 2] con todas las coordenadas x,y.
        N = body_kpts + 2*hand_kpts
        """
        print(kpts.shape)
        assert kpts.shape[1] == self.body_kpts + 2*self.hand_kpts
        G = nx.Graph()
        
        # nodos
        for i, (x,y) in enumerate(kpts.tolist()):
            G.add_node(i, x=x, y=y)
        
        # cuerpo
        G.add_edges_from(self.BODY_EDGES)
        
        # mano izquierda
        left_hand_offset = 70
        lh_edges = [(6, left_hand_offset)] + [
            (u + left_hand_offset, v + left_hand_offset)  for u, v in self.HAND_TEMPLATE
        ]
        G.add_edges_from(lh_edges)

        # Elimina conexiones de mano derecha que no deberían existir
        right_hand_offset = 90
        if G.degree[right_hand_offset] > 0:
            for neighbor in list(G.neighbors(right_hand_offset)):
                G.remove_edge(right_hand_offset, neighbor)

        # mano derecha
        rh_edges = [(3, right_hand_offset)] + [
            (u + right_hand_offset, v + right_hand_offset) for u, v in self.HAND_TEMPLATE
        ]
        G.add_edges_from(rh_edges)

        if return_adjacency:
            A = nx.adjacency_matrix(G).todense()
            return G, A

        return G, None