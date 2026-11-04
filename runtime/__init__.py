import importlib

from .runtime_observer import RuntimeObserver, RuntimeSample
from .ratio_policy import BandwidthBalancePolicy, RatioPolicy
from .ratio_controller import ControllerStatus

__all__ = [
    "BandwidthBalancePolicy",
    "ControllerStatus",
    "RatioController",
    "RatioHandle",
    "RatioPolicy",
    "Runtime",
    "RuntimeObserver",
    "RuntimeSample",
]


_EXPORTS_LOADED_ON_FIRST_USE = {
    "RatioController": ".ratio_controller",
    "Runtime": ".runtime",
    "RatioHandle": ".runtime",
}


def __getattr__(name: str):
    if name not in _EXPORTS_LOADED_ON_FIRST_USE:
        raise AttributeError(name)
    return getattr(importlib.import_module(_EXPORTS_LOADED_ON_FIRST_USE[name], __name__), name)
