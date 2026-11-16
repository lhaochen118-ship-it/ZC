#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


@dataclass(frozen=True)
class JoinCandidate:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    value_containment: float
    target_uniqueness: float
    source_non_null: float
    target_non_null: float
    source_distinct: int
    target_distinct: int
    expected_fanout: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kind(s) -> str:
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype
    if is_bool_dtype(s): return 'bool'
    if is_numeric_dtype(s): return 'number'
    if is_datetime64_any_dtype(s): return 'date'
    return 'string'


def _values(s, cap: int):
    import pandas as pd
    nonnull = s.dropna()
    if len(nonnull) > cap:
        positions = [int(i * (len(nonnull) - 1) / (cap - 1)) for i in range(cap)]
        nonnull = nonnull.iloc[positions]
    if _kind(s) == 'string':
        return [x for x in (str(v).strip() for v in nonnull) if x]
    if _kind(s) == 'date':
        return [pd.Timestamp(v).isoformat() for v in nonnull]
    if _kind(s) == 'number':
        try: return [str(Decimal(str(v)).normalize()) for v in nonnull]
        except InvalidOperation: return [str(v) for v in nonnull]
    return [str(v) for v in nonnull]


def infer_join_candidates(tables: Mapping[str, Any], *,
                          max_tables: int = 24, max_columns: int = 35,
                          max_sample: int = 3000,
                          min_support: int = 3,
                          min_containment: float = 0.80,
                          min_unique_target: float = 0.90,
                          max_edges: int = 8) -> list[JoinCandidate]:
    if max_sample < 2: raise ValueError('max_sample must be >=2')
    if min_support < 2: raise ValueError('min_support must be >=2')
    tables = {str(k): v for k, v in list(tables.items())[:max_tables]
              if hasattr(v, 'columns') and hasattr(v, '__len__') and len(v) > 0}
    profiles = {}
    for name, frame in tables.items():
        cols = list(frame.columns)[:max_columns]
        for col in cols:
            s = frame[col]
            if not hasattr(s, 'dropna') or not hasattr(s, 'dtype'): continue
            kind = _kind(s)
            if kind == 'bool': continue
            vv = _values(s, max_sample)
            if len(vv) < min_support: continue
            distinct = set(vv)
            if len(distinct) < min_support: continue
            profiles[(name, str(col))] = {
                'kind': kind, 'values': distinct, 'non_null': float(s.notna().mean()),
                'distinct': len(distinct), 'count': len(vv),
                'uniqueness': len(distinct) / len(vv),
            }
    candidates = []
    keys = sorted(profiles)
    for left, right in combinations(keys, 2):
        if left[0] == right[0]: continue
        a, b = profiles[left], profiles[right]
        if a['kind'] != b['kind']: continue
        overlap = len(a['values'] & b['values'])
        if overlap < min_support: continue
        for src_key, dst_key, src, dst in ((left, right, a, b),(right,left,b,a)):
            contain = overlap / src['distinct']
            if contain < min_containment or dst['uniqueness'] < min_unique_target: continue
            fanout = dst['count'] / max(dst['distinct'], 1)
            conf = contain * dst['uniqueness'] * math.sqrt(src['non_null'] * dst['non_null'])
            candidates.append(JoinCandidate(
                src_key[0],src_key[1],dst_key[0],dst_key[1],
                round(contain,4),round(dst['uniqueness'],4),
                round(src['non_null'],4),round(dst['non_null'],4),
                src['distinct'],dst['distinct'],round(fanout,3),round(conf,4)))
    return sorted(candidates, key=lambda c:(-c.confidence,c.source_table,c.source_column,
                                             c.target_table,c.target_column))[:max_edges]


def graph_context(tables: Mapping[str, Any], *, max_edges: int = 5) -> str:
    edges = infer_join_candidates(tables,max_edges=max_edges)
    if not edges: return ''
    lines = ['# POSSIBLE TABLE RELATIONSHIPS (STATISTICAL EVIDENCE ONLY)',
             'These links are UNSUPERVISED CANDIDATES, not guaranteed keys or business rules.',
             'Do not use a link unless the current schema, question and rulebook justify it.']
    for e in edges:
        lines.append(f'- {e.source_table}.{e.source_column} may reference '
                     f'{e.target_table}.{e.target_column} '
                     f'(overlap={e.value_containment:.2f}, target_uniqueness={e.target_uniqueness:.2f}, '
                     f'expected_fanout={e.expected_fanout:.2f})')
    return '\n'.join(lines)


@dataclass(frozen=True)
class JoinPath:
    tables: tuple[str, ...]
    links: tuple[tuple[str,str,str,str], ...]
    confidence: float
    estimated_cost: float

    def as_dict(self) -> dict[str, Any]: return asdict(self)


def candidate_join_paths(edges: list[JoinCandidate], start: str, end: str,
                         *, max_hops: int = 4, max_paths: int = 3) -> list[JoinPath]:
    import heapq
    if start == end: return [JoinPath((start,),(),1.,0.)]
    adjacency={}
    for e in edges:
        adjacency.setdefault(e.source_table,[]).append((e.target_table,e,False))
        adjacency.setdefault(e.target_table,[]).append((e.source_table,e,True))
    heap=[(0.,(start,),())]
    out=[]
    while heap and len(out)<max_paths:
        cost, path, links=heapq.heappop(heap)
        node=path[-1]
        if node==end:
            out.append(JoinPath(path,links,round(math.exp(-cost),4),round(cost,4)))
            continue
        if len(links)>=max_hops: continue
        for nxt,e,rev in adjacency.get(node,[]):
            if nxt in path:continue
            detail=((e.target_table,e.target_column,e.source_table,e.source_column) if rev
                    else (e.source_table,e.source_column,e.target_table,e.target_column))
            edge_cost=-math.log(max(e.confidence,0.001)) + (0.25 if rev else 0.)
            heapq.heappush(heap,(cost+edge_cost,path+(nxt,),links+(detail,)))
    return out
