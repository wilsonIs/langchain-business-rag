from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_question_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索前的问题改写助手。请结合对话历史，把当前用户问题改写成一个独立、清晰、可检索的问题。"
                "只补全代词指代和省略信息，不要回答问题，不要添加对话中未出现的事实。",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "当前问题：{question}"),
        ]
    )


def build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企业知识库问答助手。你必须严格基于提供的上下文回答，不能使用上下文之外的常识、训练知识或猜测。"
                "如果上下文不足以支持答案，直接回答“我不知道”。"
                "如果回答中包含结论，请在相关句子后附加 [source_id] 形式的引用标记。"
                "每个结论都必须绑定到上下文中的 source_id；若回答“我不知道”，citations 返回空数组。",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "用户问题：{question}\n\n"
                "可用上下文：\n{context}\n\n"
                "请输出结构化结果，且 citations 里的 source_id 必须原样引用上下文中的 source_id。"
                "answer 字段中的引用标记也必须使用原样 source_id。",
            ),
        ]
    )


def build_stream_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企业知识库问答助手。你必须严格基于提供的上下文回答，不能使用上下文之外的常识、训练知识或猜测。"
                "如果上下文不足以支持答案，直接回答“我不知道”。"
                "如果回答中包含结论，请在相关句子后附加 [source_id] 形式的引用标记。"
                "每个关键结论都必须绑定到上下文中的 source_id。"
                "请直接输出最终答案正文，不要输出 JSON，不要解释你的推理过程。",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "用户问题：{question}\n\n"
                "可用上下文：\n{context}\n\n"
                "请直接输出最终答案正文。若有结论，请用 [source_id] 标记来源；若证据不足，只输出“我不知道”。",
            ),
        ]
    )


def build_stream_finalize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企业知识库问答结果校验助手。你会收到用户问题、检索上下文和一段已经生成的回答草稿。"
                "你的任务是判断这段草稿是否被上下文直接支持，并输出结构化结果。"
                "如果草稿被上下文支持，请尽量保留原回答，只在必要时补上 [source_id] 引用标记。"
                "如果草稿不被上下文支持，answer 必须改为“我不知道”，grounded 为 false，citations 为空数组。"
                "citations 里的 source_id 必须严格来自上下文中的 source_id。",
            ),
            (
                "human",
                "用户问题：{question}\n\n"
                "回答草稿：{draft_answer}\n\n"
                "可用上下文：\n{context}\n\n"
                "请输出结构化结果。若草稿成立，answer 字段应尽量保持原回答表达；若草稿不成立，返回“我不知道”。",
            ),
        ]
    )
