#!/usr/bin/env ipython
from typing import Any, List


def is_not_empty(_list: List[Any]) -> bool:
    return len(_list) > 0


def tolist(tensor) -> List:
    """Tensor to python list object."""
    return tensor.cpu().detach().tolist()
