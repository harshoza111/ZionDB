from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FilterExpression(ABC):
    """Abstract base class for all metadata filter expression nodes."""

    @abstractmethod
    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        """
        Evaluates the filter expression against a document's metadata.

        Args:
            metadata: The document metadata dictionary.

        Returns:
            bool: True if the metadata satisfies the filter expression.
        """
        pass


@dataclass(frozen=True, slots=True)
class EqualFilter(FilterExpression):
    """Filter that checks if a metadata field equals a given value."""
    key: str
    value: Any

    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return False
        return metadata.get(self.key) == self.value


@dataclass(frozen=True, slots=True)
class AndFilter(FilterExpression):
    """Logical AND of a list of filter expressions."""
    expressions: List[FilterExpression]

    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        if not self.expressions:
            return True
        return all(expr.evaluate(metadata) for expr in self.expressions)


@dataclass(frozen=True, slots=True)
class OrFilter(FilterExpression):
    """Logical OR of a list of filter expressions."""
    expressions: List[FilterExpression]

    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        if not self.expressions:
            return False
        return any(expr.evaluate(metadata) for expr in self.expressions)


@dataclass(frozen=True, slots=True)
class NotFilter(FilterExpression):
    """Logical NOT of a filter expression."""
    expression: FilterExpression

    def evaluate(self, metadata: Dict[str, Any]) -> bool:
        return not self.expression.evaluate(metadata)


def parse_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional[FilterExpression]:
    """
    Parses a query filter dictionary into a FilterExpression AST.

    Supports:
    - Implicit AND for key-value pairs: {"key1": "val1", "key2": "val2"}
    - Explicit logical operators: {"$and": [...]}, {"$or": [...]}, {"$not": {...}}
    - Explicit comparison operators (e.g. {"key": {"$eq": "val"}})

    Args:
        filter_dict: The query filter dictionary, or None.

    Returns:
        Optional[FilterExpression]: The parsed AST, or None if the input is empty or None.
    """
    if not filter_dict:
        return None

    # Handle explicit logical operators if they are the only keys in the dict
    if len(filter_dict) == 1:
        key = list(filter_dict.keys())[0]
        val = filter_dict[key]

        if key == "$and":
            if not isinstance(val, list):
                raise ValueError("$and operator requires a list of filter dictionaries")
            sub_exprs = [parse_filter(f) for f in val]
            valid_exprs = [e for e in sub_exprs if e is not None]
            return AndFilter(valid_exprs) if valid_exprs else None

        elif key == "$or":
            if not isinstance(val, list):
                raise ValueError("$or operator requires a list of filter dictionaries")
            sub_exprs = [parse_filter(f) for f in val]
            valid_exprs = [e for e in sub_exprs if e is not None]
            return OrFilter(valid_exprs) if valid_exprs else None

        elif key == "$not":
            if not isinstance(val, dict):
                raise ValueError("$not operator requires a filter dictionary")
            sub_expr = parse_filter(val)
            if sub_expr is None:
                raise ValueError("$not operator cannot be empty")
            return NotFilter(sub_expr)

    # Standard key-value parsing (with implicit AND if multiple keys exist)
    expressions: List[FilterExpression] = []
    for k, v in filter_dict.items():
        # Check if value is a comparison dictionary like {"$eq": "val"}
        if isinstance(v, dict) and len(v) == 1:
            op = list(v.keys())[0]
            op_val = v[op]
            if op == "$eq":
                expressions.append(EqualFilter(k, op_val))
            elif op == "$ne":
                expressions.append(NotFilter(EqualFilter(k, op_val)))
            else:
                raise ValueError(f"Unsupported filter operator: '{op}'")
        else:
            expressions.append(EqualFilter(k, v))

    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return AndFilter(expressions)
