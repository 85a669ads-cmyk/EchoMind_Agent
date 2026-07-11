"""
长期记忆管理器 (Long-Term Memory)

封装 DashVector 向量数据库，提供语义检索、存储、更新、删除功能。
支持基于艾宾浩斯遗忘曲线的低价值记忆清理。

设计参考: 计划.md §3 模块一 - 记忆分层
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from app.models.memory_models import (
    LongTermMemoryItem,
    MemorySearchResult,
    MemoryPolarity,
)


class LongTermMemoryManager:
    """
    长期记忆管理器

    封装 DashVector SDK，在本地开发模式下回退到简易内存向量存储。

    特性:
    - 语义相似度检索（基于向量）
    - 元数据过滤（用户ID、标签、类别）
    - 支持 CRUD 操作
    - 遗忘分数低于阈值的记忆自动清理
    """

    # 默认配置
    DEFAULT_COLLECTION_NAME = "echomind_long_term_memory"
    DEFAULT_VECTOR_DIMENSION = 1536  # Qwen 嵌入模型维度

    def __init__(
        self,
        dashvector_api_key: Optional[str] = None,
        dashvector_cluster_endpoint: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_dimension: int = DEFAULT_VECTOR_DIMENSION,
        embedding_func: Optional[callable] = None,
        use_local_fallback: bool = False,
    ):
        """
        Args:
            dashvector_api_key: DashVector API Key（本地 fallback 时可为 None）
            dashvector_cluster_endpoint: DashVector 集群端点
            collection_name: 集合名称
            vector_dimension: 向量维度
            embedding_func: 文本转向量的函数，签名为 (text: str) -> list[float]
            use_local_fallback: 是否使用本地内存存储（开发/测试模式）
        """
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.embedding_func = embedding_func
        self.use_local_fallback = use_local_fallback

        # 尝试初始化 DashVector 客户端
        self._dashvector_client = None
        self._collection = None

        if not use_local_fallback:
            try:
                import dashvector
                api_key = dashvector_api_key or os.getenv("DASHVECTOR_API_KEY", "")
                endpoint = dashvector_cluster_endpoint or os.getenv("DASHVECTOR_ENDPOINT", "")

                if api_key and endpoint:
                    self._dashvector_client = dashvector.Client(
                        api_key=api_key,
                        endpoint=endpoint,
                    )
                    self._init_collection()
            except ImportError:
                print("[EchoMind] dashvector 未安装，回退到本地内存存储模式")
                self.use_local_fallback = True
            except Exception as e:
                print(f"[EchoMind] DashVector 初始化失败: {e}，回退到本地模式")
                self.use_local_fallback = True

        # 本地回退存储: { memory_id: (vector, metadata_dict) }
        self._local_store: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._local_id_counter = 0

    def _init_collection(self) -> None:
        """初始化 DashVector 集合"""
        if not self._dashvector_client:
            return

        try:
            # 获取或创建集合
            self._collection = self._dashvector_client.get(self.collection_name)
            if self._collection is None:
                from dashvector import CollectionInfo
                self._collection = self._dashvector_client.create(
                    name=self.collection_name,
                    dimension=self.vector_dimension,
                    metric="cosine",
                )
                print(f"[EchoMind] DashVector 集合 '{self.collection_name}' 已创建")
            else:
                print(f"[EchoMind] DashVector 集合 '{self.collection_name}' 已就绪")
        except Exception as e:
            print(f"[EchoMind] 集合初始化异常: {e}")
            self.use_local_fallback = True

    def _get_embedding(self, text: str) -> list[float]:
        """
        获取文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            list[float]: 向量表示
        """
        if self.embedding_func:
            return self.embedding_func(text)

        if not self.use_local_fallback:
            # 尝试使用 DashScope 嵌入
            try:
                import dashscope
                from http import HTTPStatus

                resp = dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v2,
                    input=text,
                )
                if resp.status_code == HTTPStatus.OK:
                    return resp.output["embeddings"][0]["embedding"]
            except Exception:
                pass

        # 最终回退: 使用简易 hash 向量（仅用于本地测试，不保证语义正确性）
        import hashlib
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        # 将 hash 映射到 vector_dimension 维
        vector = []
        for i in range(self.vector_dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            vector.append((byte_val / 255.0) * 2.0 - 1.0)
        # 归一化
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_memory(self, memory: LongTermMemoryItem) -> str:
        """
        添加一条长期记忆

        Args:
            memory: 长期记忆项

        Returns:
            str: 记忆ID
        """
        # 生成唯一ID
        if not memory.memory_id:
            memory.memory_id = str(uuid.uuid4())

        # 获取向量嵌入
        vector = memory.vector or self._get_embedding(memory.content)
        memory.vector = vector

        metadata = memory.to_metadata()

        if self._collection and not self.use_local_fallback:
            try:
                # 向 DashVector 插入向量
                self._collection.insert(
                    id=memory.memory_id,
                    vector=vector,
                    fields=metadata,
                )
                return memory.memory_id
            except Exception as e:
                print(f"[EchoMind] DashVector 插入失败，回退本地: {e}")

        # 本地回退
        self._local_store[memory.memory_id] = (vector, metadata)
        return memory.memory_id

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        category_filter: Optional[str] = None,
    ) -> list[MemorySearchResult]:
        """
        语义检索长期记忆

        Args:
            query: 查询文本
            user_id: 用户ID过滤（None 表示不过滤）
            top_k: 返回条数上限
            similarity_threshold: 相似度阈值（低于此值的结果被过滤）
            category_filter: 类别过滤（如 'preference'）

        Returns:
            list[MemorySearchResult]: 检索结果，按相似度降序排列
        """
        query_vector = self._get_embedding(query)

        if self._collection and not self.use_local_fallback:
            try:
                # 使用 DashVector 向量检索
                filter_expr = None
                if user_id:
                    filter_expr = f"user_id == \"{user_id}\""
                if category_filter and filter_expr:
                    filter_expr += f" and category == \"{category_filter}\""
                elif category_filter:
                    filter_expr = f"category == \"{category_filter}\""

                results = self._collection.query(
                    vector=query_vector,
                    topk=top_k,
                    filter=filter_expr,
                    include_vector=True,
                )

                search_results: list[MemorySearchResult] = []
                for i, doc in enumerate(results):
                    if doc.score >= similarity_threshold:
                        memory = LongTermMemoryItem.from_metadata(
                            memory_id=doc.id,
                            metadata=doc.fields,
                            vector=doc.vector if hasattr(doc, 'vector') else None,
                        )
                        search_results.append(MemorySearchResult(
                            memory=memory,
                            similarity_score=doc.score,
                            rank=i + 1,
                        ))
                return search_results
            except Exception as e:
                print(f"[EchoMind] DashVector 检索失败，回退本地: {e}")

        # 本地回退：暴力计算余弦相似度 + 文本片段重叠加成
        scored: list[tuple[float, str, list[float], dict]] = []
        query_lower = query.lower() if query else ""

        for mem_id, (vec, meta) in self._local_store.items():
            # 用户过滤
            if user_id and meta.get("user_id") != user_id:
                continue
            # 类别过滤
            if category_filter and meta.get("category") != category_filter:
                continue
            sim = self._cosine_similarity(query_vector, vec)

            # 文本片段重叠加成（本地模式补偿 hash 向量无语义的缺陷）
            # 兼容中英文：检查 query 是否为 content 的子串，或 content 包含 query 中的词
            content = meta.get("content", "")
            content_lower = content.lower() if content else ""

            if query_lower and content_lower:
                # 直接子串匹配
                if query_lower in content_lower or content_lower in query_lower:
                    sim = max(sim, 0.75)
                else:
                    # 逐个字符（词）的重叠度
                    query_chars = set(query_lower)
                    content_chars = set(content_lower)
                    if query_chars:
                        overlap = len(query_chars & content_chars)
                        total = min(len(query_chars), len(content_chars))
                        char_ratio = overlap / max(total, 1)
                        if char_ratio > 0.3:
                            sim = max(sim, char_ratio * 0.9)

            if sim >= similarity_threshold:
                scored.append((sim, mem_id, vec, meta))

        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        search_results: list[MemorySearchResult] = []
        for rank, (sim, mem_id, vec, meta) in enumerate(scored, 1):
            memory = LongTermMemoryItem.from_metadata(
                memory_id=mem_id,
                metadata=meta,
                vector=vec,
            )
            search_results.append(MemorySearchResult(
                memory=memory,
                similarity_score=sim,
                rank=rank,
            ))
        return search_results

    def get_by_id(self, memory_id: str) -> Optional[LongTermMemoryItem]:
        """根据ID获取记忆"""
        if self._collection and not self.use_local_fallback:
            try:
                doc = self._collection.get(memory_id)
                if doc:
                    return LongTermMemoryItem.from_metadata(
                        memory_id=doc.id,
                        metadata=doc.fields,
                    )
            except Exception:
                pass

        # 本地回退
        if memory_id in self._local_store:
            vec, meta = self._local_store[memory_id]
            return LongTermMemoryItem.from_metadata(
                memory_id=memory_id,
                metadata=meta,
                vector=vec,
            )
        return None

    def update_memory(self, memory: LongTermMemoryItem) -> bool:
        """
        更新长期记忆

        Args:
            memory: 已修改的记忆项（必须包含有效的 memory_id）

        Returns:
            bool: 是否更新成功
        """
        if not memory.memory_id:
            return False

        memory.record_access()
        vector = memory.vector or self._get_embedding(memory.content)
        memory.vector = vector
        metadata = memory.to_metadata()

        if self._collection and not self.use_local_fallback:
            try:
                self._collection.update(
                    id=memory.memory_id,
                    vector=vector,
                    fields=metadata,
                )
                return True
            except Exception as e:
                print(f"[EchoMind] DashVector 更新失败: {e}")

        # 本地回退
        self._local_store[memory.memory_id] = (vector, metadata)
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """删除一条长期记忆"""
        if self._collection and not self.use_local_fallback:
            try:
                self._collection.delete(memory_id)
                return True
            except Exception:
                pass

        if memory_id in self._local_store:
            del self._local_store[memory_id]
            return True
        return False

    def get_user_memories(
        self, user_id: str, limit: int = 100
    ) -> list[LongTermMemoryItem]:
        """获取某用户的所有长期记忆"""
        memories: list[LongTermMemoryItem] = []

        if self._collection and not self.use_local_fallback:
            try:
                results = self._collection.query(
                    topk=limit,
                    filter=f'user_id == "{user_id}"',
                    include_vector=True,
                )
                for doc in results:
                    memories.append(LongTermMemoryItem.from_metadata(
                        memory_id=doc.id,
                        metadata=doc.fields,
                        vector=doc.vector if hasattr(doc, 'vector') else None,
                    ))
                return memories
            except Exception:
                pass

        # 本地回退
        for mem_id, (vec, meta) in self._local_store.items():
            if meta.get("user_id") == user_id:
                memories.append(LongTermMemoryItem.from_metadata(
                    memory_id=mem_id,
                    metadata=meta,
                    vector=vec,
                ))
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def get_stats(self) -> dict:
        """获取长期记忆统计"""
        total = 0
        users: set[str] = set()
        total_importance = 0.0

        if self._collection and not self.use_local_fallback:
            try:
                # DashVector 可能不直接支持 count，用扫描估算
                pass
            except Exception:
                pass

        # 本地回退统计
        for mem_id, (vec, meta) in self._local_store.items():
            total += 1
            if "user_id" in meta:
                users.add(meta["user_id"])
            total_importance += float(meta.get("importance", 0.5))

        avg_importance = total_importance / max(total, 1)

        return {
            "total_long_term_memories": total,
            "active_users": len(users),
            "average_importance": round(avg_importance, 3),
        }

    def purge_low_value_memories(
        self, user_id: Optional[str] = None, forgetting_threshold: float = 0.1
    ) -> int:
        """
        清除低价值记忆（基于艾宾浩斯遗忘曲线）

        Args:
            user_id: 限定用户（None 表示全部用户）
            forgetting_threshold: 遗忘分数阈值，低于此值的记忆被删除

        Returns:
            int: 清除的记忆条数
        """
        purged = 0
        to_delete: list[str] = []

        if self._collection and not self.use_local_fallback:
            try:
                # 分批扫描 DashVector
                results = self._collection.query(
                    topk=1000,
                    filter=f'user_id == "{user_id}"' if user_id else None,
                    include_vector=False,
                )
                for doc in results:
                    memory = LongTermMemoryItem.from_metadata(
                        memory_id=doc.id,
                        metadata=doc.fields,
                    )
                    if memory.forgetting_score() < forgetting_threshold:
                        to_delete.append(doc.id)

                for mem_id in to_delete:
                    try:
                        self._collection.delete(mem_id)
                        purged += 1
                    except Exception:
                        pass
                return purged
            except Exception:
                pass

        # 本地回退
        for mem_id in list(self._local_store.keys()):
            vec, meta = self._local_store[mem_id]
            if user_id and meta.get("user_id") != user_id:
                continue
            memory = LongTermMemoryItem.from_metadata(
                memory_id=mem_id,
                metadata=meta,
            )
            if memory.forgetting_score() < forgetting_threshold:
                del self._local_store[mem_id]
                purged += 1

        return purged