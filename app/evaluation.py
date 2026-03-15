from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Dict, List

from datasets import Dataset
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings
from ragas import __version__ as ragas_version
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .embeddings import EmbeddingService
from .models import EvaluationCaseResult, EvaluationMetric, EvaluationResponse
from .rag_chain import RAGService
from .session_manager import SessionState


@dataclass(frozen=True)
class BenchmarkCase:
    question: str
    reference_answer: str


class EmbeddingAdapter(Embeddings):
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_service.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_service.embed_query(text)


class RagasEvaluationService:
    REQUIRED_SOURCES = {
        "customer_service_playbook.txt",
        "openapi_integration_runbook.txt",
        "refund_approval_policy.txt",
    }
    METRIC_LABELS = {
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "context_recall": "上下文召回",
        "context_precision": "上下文精度",
        "answer_correctness": "答案正确性",
    }
    BENCHMARK_CASES = [
        BenchmarkCase(
            question="退款金额高于 200 元时，需要谁二次确认？",
            reference_answer="退款金额高于 200 元时，必须由运营管理员二次确认后才能完成审批。",
        ),
        BenchmarkCase(
            question="夜间无人值守时系统会如何处理会话？",
            reference_answer="夜间无人值守时，系统会切换到机器人兜底流程，并保留最近十轮对话作为上下文，便于第二天人工接手。",
        ),
        BenchmarkCase(
            question="回调接口连续三次失败后，平台会怎么处理？",
            reference_answer="若回调接口连续三次失败，系统会把事件记录到告警中心，并通知运营管理员排查。",
        ),
        BenchmarkCase(
            question="正式上线前必须逐项确认哪些检查项？",
            reference_answer="正式上线前需要逐项确认白名单、签名、时钟同步、异常告警、日志脱敏和回调幂等。",
        ),
        BenchmarkCase(
            question="机器人在什么情况下必须立即转人工？",
            reference_answer="当机器人连续两次未命中答案，或者用户输入包含“投诉”“人工”“退款失败”“我要升级处理”等关键词时，系统必须立即转人工。",
        ),
    ]

    def __init__(self, rag_service: RAGService, embedding_service: EmbeddingService):
        self.rag_service = rag_service
        self.embedding_service = embedding_service

    def run_sample_benchmark(self, session: SessionState) -> EvaluationResponse:
        self._ensure_sample_docs_loaded(session)

        dataset_rows = []
        case_payloads = []
        for case in self.BENCHMARK_CASES:
            isolated_session = self._build_isolated_session(session)
            response = self.rag_service.ask(isolated_session, case.question)
            dataset_rows.append(
                {
                    "question": case.question,
                    "answer": response.answer,
                    "contexts": [document.content for document in response.source_documents],
                    "ground_truth": case.reference_answer,
                }
            )
            case_payloads.append((case, response))

        result = evaluate(
            Dataset.from_list(dataset_rows),
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness,
            ],
            llm=LangchainLLMWrapper(self.rag_service._build_llm()),
            embeddings=LangchainEmbeddingsWrapper(EmbeddingAdapter(self.embedding_service)),
            raise_exceptions=False,
        )

        case_scores = result.scores.to_list()
        summary_metrics = self._build_metric_list(dict(result))
        case_results: List[EvaluationCaseResult] = []
        for index, (case, response) in enumerate(case_payloads):
            case_results.append(
                EvaluationCaseResult(
                    question=case.question,
                    reference_answer=case.reference_answer,
                    answer=response.answer,
                    grounded=response.grounded,
                    rewritten_question=response.rewritten_question,
                    citations=response.citations,
                    source_documents=response.source_documents,
                    metrics=self._build_metric_list(case_scores[index]),
                )
            )

        return EvaluationResponse(
            benchmark_name="样例知识库 RAGAS Benchmark",
            session_id=session.session_id,
            ragas_version=ragas_version,
            sample_count=len(case_results),
            summary_metrics=summary_metrics,
            cases=case_results,
            notes=[
                "该 Benchmark 基于内置业务样例知识库运行，适合回归验证检索、重排与回答链路。",
                "RAGAS 在当前项目中评估的是单轮问答质量，不会复用当前会话历史。",
            ],
        )

    def run_sample_benchmark_threadsafe(self, session: SessionState) -> EvaluationResponse:
        try:
            asyncio.get_running_loop()
            return self.run_sample_benchmark(session)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return self.run_sample_benchmark(session)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                asyncio.set_event_loop(None)
                loop.close()

    def _ensure_sample_docs_loaded(self, session: SessionState) -> None:
        source_names = {document.source_name for document in session.documents.values()}
        missing = sorted(self.REQUIRED_SOURCES - source_names)
        if missing:
            raise RuntimeError(
                "当前 RAGAS Benchmark 仅适配内置样例知识库，请先加载样例知识库。"
                f"缺少文档: {', '.join(missing)}"
            )

    @staticmethod
    def _build_isolated_session(session: SessionState) -> SessionState:
        return SessionState(
            session_id=session.session_id,
            memory=ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
            ),
            documents=dict(session.documents),
        )

    @classmethod
    def _build_metric_list(cls, score_map: Dict[str, float]) -> List[EvaluationMetric]:
        metrics: List[EvaluationMetric] = []
        for name, label in cls.METRIC_LABELS.items():
            if name not in score_map:
                continue
            metrics.append(
                EvaluationMetric(
                    name=name,
                    label=label,
                    score=cls._normalize_score(score_map[name]),
                )
            )
        return metrics

    @staticmethod
    def _normalize_score(value) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric) or math.isinf(numeric):
            return 0.0
        return round(numeric, 4)
