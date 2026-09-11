"""
The RAG chain itself.

This upgrades the original single-turn chain into a *conversational*,
history-aware, hybrid-retrieval pipeline, and returns the source chunks
alongside the answer so the UI can show citations.

Pipeline:
  1. contextualize_question: rewrites a follow-up question ("what about
     its cost?") into a standalone question using chat history, so the
     retriever isn't blind to conversational context.
  2. history_aware_retriever: retrieves chunks using the rewritten query,
     via a HYBRID retriever (BM25 keyword search + Chroma vector search
     with MMR), so both exact terms/numbers and paraphrased/semantic
     matches are covered. Pure vector search alone is the single biggest
     cause of "the right chunk never got retrieved" in small RAG projects.
  3. qa_chain: answers strictly from the retrieved chunks, with explicit
     instructions to be precise (quote exact figures) and to say plainly
     when the context is insufficient, rather than hedging or guessing.
  4. create_retrieval_chain: wires it together and returns both the
     answer and the source documents actually used.
"""

from typing import Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

from src.config import BM25_WEIGHT, MMR_FETCH_K_MULTIPLIER, RERANK_CANDIDATE_MULTIPLIER, VECTOR_WEIGHT
from src.reranker import rerank
from src.utils import logger

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might "
    "reference context in the chat history, rewrite it as a standalone "
    "question that can be understood without the chat history. Do NOT "
    "answer the question, just reformulate it if needed, otherwise "
    "return it unchanged."
)

QA_SYSTEM_PROMPT = (
    "You are a careful enterprise document assistant. Answer the user's "
    "question using the retrieved context below.\n\n"
    "How to answer:\n"
    "1. The user's question may mix a PERSONAL detail (their specific "
    "order, how many days it's been, their account) with a GENERAL policy "
    "question. The documents can never know the personal detail — that's "
    "expected and NOT a reason to refuse. Ignore the personal detail and "
    "answer the general policy/rule the question is really asking about, "
    "quoting exact numbers, dates, or timeframes from the context.\n"
    "2. Only refuse — say exactly: \"I cannot find that information in the "
    "provided documents.\" — if the context has NO relevant policy on the "
    "topic at all. If there is ANY relevant policy text, use it; do not "
    "refuse just because the user's exact personal situation isn't in the "
    "context.\n"
    "3. If the context only partially covers the question, answer the "
    "part that is supported and state plainly what is not covered.\n"
    "4. Never invent facts, dates, or order-specific details that are not "
    "in the context. Never blend information from unrelated parts of the "
    "context into one claim.\n\n"
    "Worked example:\n"
    "Question: \"When will I get my refund, it's been 5 days?\"\n"
    "Context contains: \"refunds are processed within 5-7 business days "
    "once the return is received and inspected\"\n"
    "Good answer: \"Refunds are processed within 5-7 business days after "
    "your return is received and inspected, so you may still be within "
    "that window. I don't have visibility into your specific order's "
    "status — if it's been longer than that, the policy says you can "
    "escalate to customer care.\"\n"
    "Bad answer: \"I cannot find that information in the provided "
    "documents.\" (wrong — the general policy IS in the context; only "
    "the personal order status isn't, and that's fine to note separately.)"
    "\n\nContext:\n{context}"
)


class RerankingRetriever(BaseRetriever):
    """
    Wraps a base retriever: fetches a larger candidate set, then uses a
    cross-encoder to rerank and truncate down to top_n. See src/reranker.py
    for why this two-stage "retrieve many, rerank, keep few" approach beats
    asking the base retriever for exactly top_n candidates directly.
    """

    base_retriever: Any
    top_n: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        return rerank(query, candidates, self.top_n)


def _build_retriever(
    vectorstore, all_docs: list[Document], use_hybrid: bool, candidate_k: int
):
    """
    Build the retriever: hybrid (BM25 + vector/MMR) when a corpus is
    available, falling back to plain vector search otherwise (e.g. a
    freshly cleared knowledge base, or hybrid disabled in the UI).

    candidate_k is how many chunks to fetch (>= top_k when a reranker will
    subsequently narrow them down; equal to top_k otherwise).
    """
    fetch_k = max(candidate_k * MMR_FETCH_K_MULTIPLIER, candidate_k)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": candidate_k, "fetch_k": fetch_k},
    )

    if not use_hybrid or not all_docs:
        return vector_retriever

    try:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = candidate_k
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )
    except Exception as e:  # noqa: BLE001 - degrade to vector-only, don't crash the app
        logger.warning("BM25 index build failed (%s); falling back to vector-only retrieval", e)
        return vector_retriever


def build_rag_chain(
    vectorstore,
    llm_model: str,
    temperature: float,
    top_k: int,
    all_docs: list[Document] | None = None,
    use_hybrid: bool = True,
    use_reranker: bool = False,
):
    """Construct the history-aware, hybrid-retrieval RAG chain for a given model/config."""
    llm = ChatOllama(model=llm_model, temperature=temperature)

    candidate_k = top_k * RERANK_CANDIDATE_MULTIPLIER if use_reranker else top_k
    try:
        corpus_size = vectorstore._collection.count()
        if corpus_size:
            candidate_k = min(candidate_k, corpus_size)
    except Exception:  # noqa: BLE001 - clamping is a nicety, not critical
        pass
    retriever = _build_retriever(vectorstore, all_docs or [], use_hybrid, candidate_k)
    if use_reranker:
        retriever = RerankingRetriever(base_retriever=retriever, top_n=top_k)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


def to_lc_messages(streamlit_messages: list[dict]) -> list:
    """Convert Streamlit's session_state message dicts into LangChain message objects."""
    lc_messages = []
    for m in streamlit_messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages
