import contextlib
import torch
import torch.nn as nn

from typing import *
from functools import wraps
from dataclasses import dataclass
from pydantic import BaseModel, PrivateAttr
from torch.utils.hooks import RemovableHandle

__all__ = ["HooksMixin", "PersistentHook"]

class HooksMixin(BaseModel):
    """
    Mixin to manage hook registration, disabling, and removal
    """
    _HOOKS_DISABLED: ClassVar[bool] = False
    _HOOKS_ENABLED: ClassVar[set[RemovableHandle]] = set()
    _PERSISTENT_HOOKS: ClassVar[set[RemovableHandle]] = set()

    _hooks: set[RemovableHandle] = PrivateAttr(default_factory=set)

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(
        cls, keep: set[RemovableHandle] = frozenset()
    ):
        """
        Disable all hooks except handles in `keep` and persistent hooks
        """
        try:
            cls._HOOKS_DISABLED = True
            cls._HOOKS_ENABLED |= keep | cls._PERSISTENT_HOOKS
            yield

        finally:
            cls._HOOKS_DISABLED = False
            cls._HOOKS_ENABLED -= keep
    
    def register_hook(
        self, 
        target: nn.Module | nn.Parameter,
        hook: Callable,
        hook_type: str,
        **kwargs
    ):
        """
        Registers a hook on a specified module
        """
        handle = None

        @wraps(hook)
        def wrapped_hook(*args, **kwargs):
            nonlocal handle

            if (
                HooksMixin._HOOKS_DISABLED
                and handle not in HooksMixin._HOOKS_ENABLED
            ):
                return

            return hook(*args, **kwargs)

        func = getattr(target, f"register_{hook_type}_hook")
        handle = func(wrapped_hook, **kwargs)
        self._hooks.add(handle)

        return handle

    def register_persistent_hook(
        self,
        target: nn.Module | nn.Parameter,
        hook: Callable,
        hook_type: str,
        **kwargs
    ):
        """
        Register a persistent hook kept during disable_hooks
        """
        handle = self.register_hook(target, hook, hook_type, **kwargs)
        HooksMixin._PERSISTENT_HOOKS.add(handle)
        return handle

    def remove_hooks(
        self, handles: set[RemovableHandle] | None = None
    ):
        """
        Remove hooks which are registered
        """
        if handles is None:
            handles = self._hooks
        
        for hook in handles:
            hook.remove()
        
        self._hooks -= handles

@dataclass
class PersistentHook:
    """
    Base class for persistent hooks that transform tensors
    """
    def process(self, x: torch.Tensor):
        ...

    def as_input_hook(self):
        """
        Make forward_pre_hook (with_kwargs=True)
        """
        def hook(_, args, kwargs):
            if args:
                x = args[0]
                return (self.process(x),) + args[1:], kwargs
            else:
                key = next(iter(kwargs))
                x = kwargs[key]
                return args, {**kwargs, key: self.process(x)}
        
        return hook

    def as_output_hook(self):
        """
        Make forward_hook
        """
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                return (self.process(output[0]),) + output[1:]
            return self.process(output)
        
        return hook
