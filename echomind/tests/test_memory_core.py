"""
EchoMind 2.0 记忆核心层单元测试

测试覆盖:
- MemoryItem 数据模型的遗忘分数计算
- WorkingMemory 的增删查和 TTL 过期
- LongTermMemoryManager 的 CRUD 和检索
- ConflictResolver 的冲突检测
- ForgettingEngine 的遗忘曲线计算
- 完整记忆流集成测试
"""

import time
import pytest
import asyncio

from app.models.memory_models import (
    MemoryItem,
    WorkingMemoryItem,
    LongTermMemoryItem,
    MemoryPolarity,
)
from app.memory.working_memory import WorkingMemory
from app.memory.long_term_memory import LongTermMemoryManager
from app.memory.conflict_resolver import ConflictResolver
from app.memory.forgetting_engine import ForgettingEngine


# ==========================================
# 1. MemoryItem 数据模型测试
# ==========================================

class TestMemoryItem:
    """测试记忆基类"""

    def test_create_memory_item(self):
        item = MemoryItem(content="测试记忆", importance=0.8, polarity=MemoryPolarity.POSITIVE)
        assert item.content == "测试记忆"
        assert item.importance == 0.8
        assert item.polarity == MemoryPolarity.POSITIVE
        assert item.access_count == 0

    def test_record_access(self):
        item = MemoryItem(content="测试")
        assert item.access_count == 0
        item.record_access()
        assert item.access_count == 1
        assert item.last_accessed_at is not None

    def test_forgetting_score_high_importance(self):
        item = MemoryItem(content="重要记忆", importance=0.9)
        item.record_access()
        score = item.forgetting_score()
        assert score > 0.5  # 新记忆 + 高重要性 = 高分

    def test_forgetting_score_low_importance(self):
        item = MemoryItem(content="无关紧要", importance=0.1)
        score = item.forgetting_score()
        assert score < 0.5

    def test_forgetting_score_decay_over_time(self):
        old_item = MemoryItem(
            content="旧记忆",
            importance=0.8,
            created_at=time.time() - 3600 * 24 * 7,  # 7天前
        )
        new_item = MemoryItem(content="新记忆", importance=0.8)
        assert old_item.forgetting_score() < new_item.forgetting_score()

    def test_forgetting_score_frequent_access_bonus(self):
        frequent_item = MemoryItem(content="常用记忆", importance=0.5)
        for _ in range(100):
            frequent_item.record_access()
        rare_item = MemoryItem(content="很少访问", importance=0.5)
        assert frequent_item.forgetting_score() > rare_item.forgetting_score()


# ==========================================
# 2. WorkingMemoryItem 测试
# ==========================================

class TestWorkingMemoryItem:
    """测试短期工作记忆"""

    def test_create_working_memory(self):
        item = WorkingMemoryItem(
            user_id="user_001",
            session_id="session_001",
            content="用户消息",
        )
        assert item.user_id == "user_001"
        assert item.session_id == "session_001"
        assert item.ttl_seconds == 1800

    def test_is_expired(self):
        item = WorkingMemoryItem(
            user_id="user_001",
            session_id="session_001",
            content="消息",
            ttl_seconds=1,
        )
        assert not item.is_expired()
        time.sleep(1.1)
        assert item.is_expired()


# ==========================================
# 3. WorkingMemory 管理器测试
# ==========================================

class TestWorkingMemory:
    """测试短期工作记忆管理器"""

    def setup_method(self):
        self.wm = WorkingMemory(
            max_items_per_user=50,
            default_ttl_seconds=3600,
            consolidate_threshold=10,
        )

    def test_add_and_get_recent(self):
        self.wm.add("user_001", "消息1")
        self.wm.add("user_001", "消息2")
        self.wm.add("user_001", "消息3")
        recent = self.wm.get_recent("user_001")
        assert len(recent) == 3

    def test_count(self):
        self.wm.add("user_001", "消息1")
        self.wm.add("user_001", "消息2")
        assert self.wm.count("user_001") == 2

    def test_get_recent_limits(self):
        for i in range(15):
            self.wm.add("user_001", f"消息{i}")
        recent = self.wm.get_recent("user_001", n=5)
        assert len(recent) == 5

    def test_clear_session(self):
        self.wm.add("user_001", "消息1", session_id="sess_1")
        self.wm.add("user_001", "消息2", session_id="sess_2")
        self.wm.clear_session("user_001", "sess_1")
        assert self.wm.count("user_001", "sess_1") == 0
        assert self.wm.count("user_001", "sess_2") == 1

    def test_clear_user(self):
        self.wm.add("user_001", "消息1")
        self.wm.add("user_001", "消息2")
        self.wm.add("user_002", "消息3")
        self.wm.clear_user("user_001")
        assert self.wm.count("user_001") == 0
        assert self.wm.count("user_002") == 1

    def test_consolidate_callback(self):
        consolidated_items = []

        def callback(user_id, items):
            consolidated_items.append((user_id, items))

        wm = WorkingMemory(consolidate_threshold=5, on_consolidate_callback=callback)
        for i in range(5):
            wm.add("user_001", f"消息{i}")
        assert len(consolidated_items) >= 1

    def test_does_not_consolidate_below_threshold(self):
        consolidated = []

        def callback(user_id, items):
            consolidated.append(items)

        wm = WorkingMemory(consolidate_threshold=10, on_consolidate_callback=callback)
        for i in range(5):
            wm.add("user_001", f"消息{i}")
        assert len(consolidated) == 0

    def test_get_stats(self):
        self.wm.add("user_001", "消息1")
        self.wm.add("user_002", "消息2")
        self.wm.add("user_002", "消息3")
        stats = self.wm.get_stats()
        assert stats["active_users"] == 2
        assert stats["total_working_memories"] == 3


# ==========================================
# 4. LongTermMemoryManager 本地回退测试
# ==========================================

class TestLongTermMemoryManager:
    """测试长期记忆管理器（本地回退模式）"""

    def setup_method(self):
        self.ltm = LongTermMemoryManager(use_local_fallback=True)

    def test_add_and_search(self):
        memory = LongTermMemoryItem(
            user_id="user_001",
            content="用户喜欢喝冰美式咖啡",
            category="preference",
            importance=0.8,
            tags=["咖啡", "偏好"],
        )
        mem_id = self.ltm.add_memory(memory)
        assert mem_id
        results = self.ltm.search(query="咖啡", user_id="user_001", top_k=5)
        assert len(results) >= 1

    def test_search_by_user_id(self):
        memory_a = LongTermMemoryItem(user_id="user_A", content="用户A的记忆")
        memory_b = LongTermMemoryItem(user_id="user_B", content="用户B的记忆")
        self.ltm.add_memory(memory_a)
        self.ltm.add_memory(memory_b)
        results = self.ltm.search(query="记忆", user_id="user_A")
        assert len(results) == 1
        assert results[0].memory.user_id == "user_A"

    def test_get_by_id(self):
        memory = LongTermMemoryItem(user_id="user_001", content="测试记忆")
        mem_id = self.ltm.add_memory(memory)
        retrieved = self.ltm.get_by_id(mem_id)
        assert retrieved is not None
        assert retrieved.content == "测试记忆"

    def test_get_by_id_not_found(self):
        result = self.ltm.get_by_id("non_existent_id")
        assert result is None

    def test_update_memory(self):
        memory = LongTermMemoryItem(user_id="user_001", content="原始内容")
        mem_id = self.ltm.add_memory(memory)
        memory.content = "更新后的内容"
        memory.memory_id = mem_id
        self.ltm.update_memory(memory)
        updated = self.ltm.get_by_id(mem_id)
        assert updated.content == "更新后的内容"

    def test_delete_memory(self):
        memory = LongTermMemoryItem(user_id="user_001", content="待删除记忆")
        mem_id = self.ltm.add_memory(memory)
        assert self.ltm.delete_memory(mem_id)
        assert self.ltm.get_by_id(mem_id) is None

    def test_get_user_memories(self):
        for i in range(5):
            memory = LongTermMemoryItem(user_id="user_001", content=f"记忆{i}")
            self.ltm.add_memory(memory)
        memories = self.ltm.get_user_memories("user_001", limit=3)
        assert len(memories) == 3
        all_memories = self.ltm.get_user_memories("user_001")
        assert len(all_memories) == 5

    def test_get_stats(self):
        memory = LongTermMemoryItem(user_id="user_001", content="测试统计", importance=0.7)
        self.ltm.add_memory(memory)
        stats = self.ltm.get_stats()
        assert stats["total_long_term_memories"] == 1
        assert stats["average_importance"] > 0


# ==========================================
# 5. ConflictResolver 测试
# ==========================================

class TestConflictResolver:
    """测试冲突解决器"""

    def setup_method(self):
        self.resolver = ConflictResolver(similarity_threshold=0.85)

    def test_no_conflict_without_ltm(self):
        memory = LongTermMemoryItem(user_id="user_001", content="新记忆")
        result = asyncio.run(self.resolver.resolve_conflict("user_001", memory))
        assert result.resolution == "keep_both"
        assert "跳过冲突检测" in result.reason


# ==========================================
# 6. ForgettingEngine 测试
# ==========================================

class TestForgettingEngine:
    """测试遗忘引擎"""

    def setup_method(self):
        self.engine = ForgettingEngine(
            long_term_memory_manager=None,
            forgetting_threshold=0.1,
            decay_rate_k=0.05,
            auto_purge=False,
        )

    def test_evaluate_high_importance_memory(self):
        memory = LongTermMemoryItem(user_id="user_001", content="重要偏好", importance=0.9)
        memory.record_access()
        score = self.engine.evaluate_memory(memory)
        assert score > 0.5

    def test_evaluate_low_importance_memory(self):
        memory = LongTermMemoryItem(
            user_id="user_001",
            content="过时信息",
            importance=0.1,
            created_at=time.time() - 3600 * 24 * 30,
        )
        score = self.engine.evaluate_memory(memory)
        assert score < 0.5

    def test_should_forget(self):
        memory = LongTermMemoryItem(
            user_id="user_001",
            content="低价值记忆",
            importance=0.05,
            created_at=time.time() - 3600 * 24 * 100,
        )
        should_del, score = self.engine.should_forget(memory)
        assert should_del or score < self.engine.forgetting_threshold

    def test_purge_without_ltm(self):
        result = self.engine.purge(user_id="user_001")
        assert "error" in result or result["total_evaluated"] == 0

    def test_get_stats(self):
        stats = self.engine.get_stats()
        assert "total_purged" in stats
        assert stats["auto_purge"] is False


# ==========================================
# 7. 集成测试：完整记忆流
# ==========================================

class TestMemoryIntegration:
    """测试记忆系统的完整流程"""

    def test_full_memory_lifecycle(self):
        ltm = LongTermMemoryManager(use_local_fallback=True)
        memory = LongTermMemoryItem(
            user_id="user_001",
            content="用户偏好：喜欢Python编程",
            category="preference",
            importance=0.8,
            tags=["编程", "Python"],
        )
        mem_id = ltm.add_memory(memory)
        assert mem_id
        results = ltm.search(query="Python", user_id="user_001")
        assert len(results) >= 1
        retrieved = ltm.get_by_id(mem_id)
        retrieved.record_access()
        ltm.update_memory(retrieved)
        updated = ltm.get_by_id(mem_id)
        assert updated.access_count >= 1
        ltm.delete_memory(mem_id)
        assert ltm.get_by_id(mem_id) is None

    def test_working_memory_overflow(self):
        wm = WorkingMemory(max_items_per_user=5)
        for i in range(10):
            wm.add("user_001", f"消息{i}")
        recent = wm.get_recent("user_001")
        assert len(recent) == 5