from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from qmshe.ingest.schemas import Entity, EvidenceHyperedge


@dataclass(frozen=True)
class IncidenceResult:
    matrix: sp.csr_matrix
    node_ids: list[str]
    edge_ids: list[str]
    edge_weights: np.ndarray


def build_incidence(
    entities: list[Entity], hyperedges: list[EvidenceHyperedge]
) -> IncidenceResult:
    node_ids = [entity.entity_id for entity in entities]
    edge_ids = [edge.hyperedge_id for edge in hyperedges]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    rows, cols = [], []
    for col, edge in enumerate(hyperedges):
        for node_id in {argument.entity_id for argument in edge.arguments}:
            if node_id in node_index:
                rows.append(node_index[node_id])
                cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(entities), len(hyperedges))).tocsr()
    weights = np.asarray([edge.confidence for edge in hyperedges], dtype=np.float32)
    return IncidenceResult(matrix=matrix, node_ids=node_ids, edge_ids=edge_ids, edge_weights=weights)


def build_role_incidence(
    entities: list[Entity], hyperedges: list[EvidenceHyperedge]
) -> dict[str, sp.csr_matrix]:
    node_index = {entity.entity_id: i for i, entity in enumerate(entities)}
    roles = sorted({argument.role for edge in hyperedges for argument in edge.arguments})
    output = {}
    for role in roles:
        rows, cols = [], []
        for col, edge in enumerate(hyperedges):
            for argument in edge.arguments:
                if argument.role == role and argument.entity_id in node_index:
                    rows.append(node_index[argument.entity_id])
                    cols.append(col)
        output[role] = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(entities), len(hyperedges)),
        ).tocsr()
    return output

