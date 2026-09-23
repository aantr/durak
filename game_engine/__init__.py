"""Движок подкидного дурака и адаптер состояния распознавания."""

# При обычной установке расширение находится в site-packages. Этот каталог
# исходников тоже является game_engine, поэтому расширяем путь пакета.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .game_engine import DurakEngine, observation_from_state

__all__ = ["DurakEngine", "observation_from_state"]
