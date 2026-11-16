#!/usr/bin/env python3
from __future__ import annotations
import ast
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Mapping, Any

_UNIT = re.compile(r'^([\w\u00c0-\u1ef9]+)(?:\^(-?\d+))?$', re.UNICODE)


class DimensionError(ValueError):
    pass


def normalize(d: Mapping[str,int]) -> dict[str,int]:
    return {str(k):int(v) for k,v in sorted(d.items()) if int(v) != 0}


def parse_dimension(expr: str) -> dict[str,int]:
    s=''.join(str(expr).split())
    if s in ('','1','dimensionless'): return {}
    if s.startswith(('*','/')) or s.endswith(('*','/')):raise DimensionError('invalid unit expression')
    pieces=re.split(r'([*/])',s)
    acc=Counter(); sign=1
    for i,part in enumerate(pieces):
        if i%2:
            if part not in ('*','/'): raise DimensionError('bad separator')
            sign = 1 if part=='*' else -1
            continue
        m=_UNIT.fullmatch(part)
        if not m: raise DimensionError('invalid unit symbol: '+part[:60])
        unit,power=m.group(1),int(m.group(2) or 1)
        if unit=='1': continue
        acc[unit]+=sign*power
    return normalize(acc)


def multiply(a: Mapping[str,int],b: Mapping[str,int], div=False) -> dict[str,int]:
    c=Counter(a)
    for k,v in b.items():c[k]+=(-1 if div else 1)*v
    return normalize(c)


def infer_expression(expr: str, units: Mapping[str,str]) -> dict[str,int]:
    src_units={str(name):parse_dimension(value) for name,value in units.items()}
    try: tree=ast.parse(str(expr),mode='eval')
    except SyntaxError as e: raise DimensionError('invalid arithmetic expression') from e
    def walk(n):
        if isinstance(n,ast.Name):
            if n.id not in src_units:raise DimensionError('unknown symbol: '+n.id)
            return src_units[n.id]
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and not isinstance(n.value,bool):return {}
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return walk(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,(ast.Add,ast.Sub)):
                if a != b:raise DimensionError('dimension mismatch in addition/subtraction')
                return a
            if isinstance(n.op,ast.Mult):return multiply(a,b)
            if isinstance(n.op,ast.Div):return multiply(a,b,True)
            if isinstance(n.op,ast.Pow):
                if b:raise DimensionError('dimensioned exponent')
                if not isinstance(n.right,ast.Constant) or not isinstance(n.right.value,int):
                    raise DimensionError('exponent must be literal integer')
                return normalize({k:v*n.right.value for k,v in a.items()})
        raise DimensionError('unsupported syntax: '+type(n).__name__)
    return walk(tree.body)


@dataclass(frozen=True)
class CertificateResult:
    valid: bool
    inferred: dict[str,int] | None
    claimed: dict[str,int] | None
    reason: str

    def as_dict(self)->dict[str,Any]:return asdict(self)


def validate_certificate(cert:Mapping[str,Any]) -> CertificateResult:
    try:
        variables=cert['variables']; expression=cert['expression']; claimed=parse_dimension(cert['result_unit'])
        if not isinstance(variables,dict) or len(variables)>32:raise DimensionError('invalid variables')
        got=infer_expression(expression,variables)
        if got!=claimed:return CertificateResult(False,got,claimed,'claimed result unit disagrees with algebra')
        return CertificateResult(True,got,claimed,'dimensionally consistent; semantics not verified')
    except (DimensionError,KeyError,TypeError,ValueError) as e:
        return CertificateResult(False,None,None,str(e))
