from typing import *
from collections import defaultdict

__all__ = ["RegistryMixin"]

_REGISTRY: Dict[type, Dict[str, Any]] = defaultdict(dict)

class RegistryMixin:
    """
    Mixin to support registration and loading 
    of child classes and plugins
    """
    @classmethod
    def register(cls, name: str | None = None):
        """
        Decorator for registering a value
        (i.e. class or function)
        """
        def decorator(value: Any):
            register(cls, value, name)
            return value
        
        return decorator
    
    @classmethod
    def get(cls, name: str):
        """
        Return retrieved value from the registry
        for the given name
        """
        return get(cls, name)

def standardize_name(name: str):
    """
    Standardize the given name in the registry
    """
    return name.replace("_", "-").replace(" ", "-").lower()

def register(cls: type, value: Any, name: str | None = None):
    if name is None:
        name = value.__name__
    name = standardize_name(name)

    # raise error if value is not subclass of cls
    if not issubclass(value, cls):
        raise ValueError(f"{value} is not a subclass of {cls}")
    
    # raise error if already registered
    if name in _REGISTRY[cls]:
        registered = _REGISTRY[cls][name]
        if registered is not value:
            raise RuntimeError(f"{name} has has already been registered as {registered}")

    # register new value to _REGISTRY
    else:
        _REGISTRY[cls][name] = value

def get(cls: type, name: str):
    name = standardize_name(name)

    # retrieve
    retrieved = _REGISTRY[cls].get(name)

    # if retrieved is a string, it's an alias - resolve it
    if isinstance(retrieved, str):
        retrieved = _REGISTRY[cls].get(retrieved)

    # raise error if not in _REGISTRY
    if retrieved is None:
        raise KeyError(f"Unable to find {name} registered under {cls}")

    # raise error if retrieved is not subclass of cls
    if not issubclass(retrieved, cls):
        raise ValueError(f"{retrieved} is not a subclass of {cls}")

    return retrieved