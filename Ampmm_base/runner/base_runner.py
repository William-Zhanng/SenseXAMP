from abc import ABCMeta, abstractmethod
from .hooks import Hook
from .priority import get_priority,Priority
from typing import Union

class BaseRunner(metaclass=ABCMeta):
    """
    Base class of runner, a tool for pytorch model training.
    """
    def __init__(self) -> None:
        self._hooks = []
        self._epoch = 0
        self._iter = 0 # total iteration nums
        self._inner_iter = 0 # iteration of each epoch

    def register_hooks(self, hooks) -> None:
        """Register hooks to the trainer.
        Args:
            hooks (list[HookBase]): List of hooks to be registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self,hook: Hook, priority:Union[int, str, Priority]='NORMAL') -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
              
    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self):
        pass