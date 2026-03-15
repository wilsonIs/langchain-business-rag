from __future__ import annotations

import re
from typing import AsyncIterator, Dict, List

from langchain_openai import ChatOpenAI

from .config import Settings
from .models import ChatResponse, Citation, SourceDocument, StructuredAnswer
from .prompts import (
    build_answer_prompt,
    build_question_rewrite_prompt,
    build_stream_finalize_prompt,
    build_stream_answer_prompt,
)
from .session_manager import SessionState
from .vector_store import VectorIndex


def format_context(source_documents: List[SourceDocument]) -> str:
    if not source_documents:
        return "没有可用上下文。"

    blocks = []
    for document in source_documents:
        blocks.append(
            f"[source_id={document.source_id} | source_name={document.source_name} | "
            f"segment={document.segment_label} | score={document.score:.4f}]\n{document.content}"
        )
    return "\n\n".join(blocks)


class RAGService:
    SOURCE_ID_PATTERN = re.compile(r"\[([^\[\]]+)\]")

    def __init__(self, settings: Settings, vector_index: VectorIndex):
        self.settings = settings
        self.vector_index = vector_index
        self._rewrite_prompt = build_question_rewrite_prompt()
        self._answer_prompt = build_answer_prompt()
        self._stream_answer_prompt = build_stream_answer_prompt()
        self._stream_finalize_prompt = build_stream_finalize_prompt()

    def ensure_ready(self, session: SessionState) -> None:
        if not session.documents:
            raise RuntimeError("当前会话还没有知识库文档，请先上传文档或加载样例知识库。")
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "未检测到模型 API Key，请先配置 DEEPSEEK_API_KEY，"
                "或在 OpenAI 兼容模式下配置 OPENAI_API_KEY。"
            )

    def _build_llm(self, streaming: bool = False):
        self.ensure_api_key()

        kwargs = {
            "model": self.settings.llm_model,
            "api_key": self.settings.llm_api_key,
            "temperature": 0,
            "streaming": streaming,
        }
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        return ChatOpenAI(**kwargs)

    def ensure_api_key(self) -> None:
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "未检测到模型 API Key，请先配置 DEEPSEEK_API_KEY，"
                "或在 OpenAI 兼容模式下配置 OPENAI_API_KEY。"
            )

    @staticmethod
    def _message_to_text(message, strip: bool = True) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip() if strip else content

        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        text = "\n".join(parts)
        return text.strip() if strip else text

    def _rewrite_question(self, llm: ChatOpenAI, chat_history, question: str) -> str:
        prompt_value = self._rewrite_prompt.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )
        response_message = llm.invoke(prompt_value)
        return self._message_to_text(response_message) or question

    async def _rewrite_question_async(self, llm: ChatOpenAI, chat_history, question: str) -> str:
        prompt_value = self._rewrite_prompt.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )
        response_message = await llm.ainvoke(prompt_value)
        return self._message_to_text(response_message) or question

    def _generate_structured_answer(
        self,
        structured_llm,
        chat_history,
        question: str,
        context: str,
    ) -> StructuredAnswer:
        prompt_value = self._answer_prompt.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "context": context,
            }
        )
        return structured_llm.invoke(prompt_value)

    def _stream_answer_prompt_value(self, chat_history, question: str, context: str):
        return self._stream_answer_prompt.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "context": context,
            }
        )

    async def _finalize_streamed_answer(
        self,
        structured_llm,
        question: str,
        draft_answer: str,
        context: str,
    ) -> StructuredAnswer:
        prompt_value = self._stream_finalize_prompt.invoke(
            {
                "question": question,
                "draft_answer": draft_answer,
                "context": context,
            }
        )
        return await structured_llm.ainvoke(prompt_value)

    def ask(self, session: SessionState, question: str) -> ChatResponse:
        self.ensure_ready(session)

        llm = self._build_llm()
        structured_llm = llm.with_structured_output(StructuredAnswer)
        chat_history = session.memory.load_memory_variables({}).get("chat_history", [])

        # Step 1: rewrite the follow-up question into a standalone query.
        standalone_question = self._rewrite_question(llm, chat_history, question)

        # Step 2: retrieve source documents from the local vector store.
        source_documents = self.vector_index.search(
            session_id=session.session_id,
            query=standalone_question,
            top_k=self.settings.top_k,
        )
        context = format_context(source_documents)

        # Step 3: ask the model to answer strictly from retrieved context.
        structured_answer = self._generate_structured_answer(
            structured_llm=structured_llm,
            chat_history=chat_history,
            question=question,
            context=context,
        )
        citations = self._sanitize_citations(structured_answer.citations, source_documents)
        answer_text = structured_answer.answer.strip() or "我不知道"
        grounded = bool(structured_answer.grounded and citations)
        if not grounded and answer_text != "我不知道":
            answer_text = "我不知道"
            citations = []

        response = ChatResponse(
            session_id=session.session_id,
            answer=answer_text,
            grounded=grounded,
            rewritten_question=standalone_question,
            citations=citations,
            source_documents=source_documents,
        )
        session.memory.save_context({"question": question}, {"answer": response.answer})
        return response

    async def stream_ask(self, session: SessionState, question: str) -> AsyncIterator[Dict[str, object]]:
        self.ensure_ready(session)

        llm = self._build_llm(streaming=True)
        chat_history = session.memory.load_memory_variables({}).get("chat_history", [])

        standalone_question = await self._rewrite_question_async(llm, chat_history, question)
        source_documents = self.vector_index.search(
            session_id=session.session_id,
            query=standalone_question,
            top_k=self.settings.top_k,
        )
        context = format_context(source_documents)

        yield {
            "event": "meta",
            "data": {
                "rewritten_question": standalone_question,
            },
        }

        prompt_value = self._stream_answer_prompt_value(
            chat_history=chat_history,
            question=question,
            context=context,
        )
        answer_parts: List[str] = []
        async for chunk in llm.astream(prompt_value):
            text = self._message_to_text(chunk, strip=False)
            if not text:
                continue
            answer_parts.append(text)
            yield {"event": "delta", "data": {"text": text}}

        answer_text = "".join(answer_parts).strip() or "我不知道"
        citations: List[Citation] = []
        grounded = False
        try:
            structured_llm = self._build_llm().with_structured_output(StructuredAnswer)
            finalized_answer = await self._finalize_streamed_answer(
                structured_llm=structured_llm,
                question=question,
                draft_answer=answer_text,
                context=context,
            )
            answer_text = finalized_answer.answer.strip() or answer_text
            citations = self._sanitize_citations(finalized_answer.citations, source_documents)
            if finalized_answer.grounded and not citations:
                citations = self._extract_citations_from_answer(answer_text, source_documents)
            grounded = bool(finalized_answer.grounded and citations)
        except Exception:
            citations = self._extract_citations_from_answer(answer_text, source_documents)
            grounded = bool(citations) and answer_text != "我不知道"

        if not grounded and answer_text != "我不知道":
            answer_text = "我不知道"
            citations = []

        response = ChatResponse(
            session_id=session.session_id,
            answer=answer_text,
            grounded=grounded,
            rewritten_question=standalone_question,
            citations=citations,
            source_documents=source_documents,
        )
        session.memory.save_context({"question": question}, {"answer": response.answer})
        yield {"event": "final", "data": response.model_dump()}

    @staticmethod
    def _sanitize_citations(
        citations: List[Citation],
        source_documents: List[SourceDocument],
    ) -> List[Citation]:
        valid_sources = {document.source_id: document for document in source_documents}
        sanitized: List[Citation] = []
        for citation in citations:
            if citation.source_id not in valid_sources:
                continue
            source_document = valid_sources[citation.source_id]
            sanitized.append(
                Citation(
                    source_id=citation.source_id,
                    source_name=source_document.source_name,
                    segment_label=source_document.segment_label,
                    supporting_text=citation.supporting_text.strip(),
                )
            )
        return sanitized

    @classmethod
    def _extract_citations_from_answer(
        cls,
        answer_text: str,
        source_documents: List[SourceDocument],
    ) -> List[Citation]:
        valid_sources = {document.source_id: document for document in source_documents}
        citations: List[Citation] = []
        seen = set()

        for source_id in cls.SOURCE_ID_PATTERN.findall(answer_text):
            normalized_source_id = source_id.strip()
            if normalized_source_id in seen or normalized_source_id not in valid_sources:
                continue
            source_document = valid_sources[normalized_source_id]
            citations.append(
                Citation(
                    source_id=normalized_source_id,
                    source_name=source_document.source_name,
                    segment_label=source_document.segment_label,
                    supporting_text=source_document.content.strip(),
                )
            )
            seen.add(normalized_source_id)
        return citations
