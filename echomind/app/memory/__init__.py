"""
EchoMind 2.0 记忆核心层
包含短期记忆、长期记忆、巩固器、冲突解决器和遗忘引擎。
"""

from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemoryManager
from .consolidator import MemoryConsolidator
from .conflict_resolver import ConflictResolver
from .forgetting_engine import ForgettingEngine

__all__ = [
    "WorkingMemory",
    "LongTermMemoryManager",
    "MemoryConsolidator",
    "ConflictResolver",
    "ForgettingEngine",
]