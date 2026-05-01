import os
import anthropic
from ingestion.image_extractor import ExtractedImage


# ── System prompts ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise document assistant with vision capabilities.
Answer the user's question using ONLY the provided document context, which may include
both text and images. Describe and reason about images when they are relevant.
If the answer is not in the context, say "I couldn't find that in the document."
Be concise and factual. Do not hallucinate or add outside knowledge."""

UNRESTRICTED_SYSTEM_PROMPT = """You are a knowledgeable and thoughtful assistant helping
a user understand a document they have uploaded.

You will be given relevant excerpts from the document alongside the user's question.
Use the document context as a starting point, then freely expand with your general
knowledge to give the most helpful, complete answer possible.

You are NOT restricted to only what the document says — you can and should add
background knowledge, explanations, comparisons, implications, and context that
helps the user understand the topic more deeply.

If the document context is not relevant to the question, answer from general knowledge."""

SYNTHESIS_SYSTEM_PROMPT = """You are a document analysis assistant helping users deeply
understand an uploaded document.

You will receive:
1. A question from the user
2. A RAG answer — pulled strictly from the document
3. An unrestricted answer — from general knowledge combined with document context

YOUR TASK — decide which of these three cases applies, then respond accordingly:

────────────────────────────────────────────────────────
CASE 1 — Question IS answered by the document
  Condition: RAG answer contains relevant content.
  Action: Lead with the document's answer. Use the unrestricted answer to add
          background, context, or explanation that helps the user understand
          the document better. Clearly label what comes from the document
          vs general knowledge.
────────────────────────────────────────────────────────
CASE 2 — Question is RELATED to the document but not explicitly answered
  Condition: RAG says "I couldn't find that in the document" BUT the question
             is about the same subject matter, domain, or topic as the document.
             Examples: asking about a concept implied by the document, asking
             for context around the document's topic, asking something the
             document touches on but doesn't detail.
  Action: Use the unrestricted answer to give a full, helpful response.
          Clearly note this comes from general knowledge, not the document itself.
────────────────────────────────────────────────────────
CASE 3 — Question is COMPLETELY UNRELATED to the document
  Condition: RAG says "I couldn't find that in the document" AND the question
             has no meaningful connection to the document's subject matter,
             domain, or topic. The question could have been asked with zero
             knowledge of the document.
             Examples: asking about movies, recipes, sports teams, or any topic
             that shares no domain with the document.
  Action: Respond ONLY with:
          "This question is not related to the uploaded document. Please ask
          something about the document's content or its subject matter."
          Do NOT use the unrestricted answer. Do NOT provide any other information.
────────────────────────────────────────────────────────

When in doubt between CASE 2 and CASE 3, favour CASE 2 and answer helpfully."""


def _build_content(
    question: str,
    context: str,
    images: list[ExtractedImage] | None = None,
) -> list[dict]:
    """Build the content list for a Claude API call with optional images."""
    content: list[dict] = []

    if images:
        content.append({
            "type": "text",
            "text": (
                f"Here is the relevant text from the document:\n\n{context}\n\n"
                f"The document also contains {len(images)} image(s) shown below. "
                "Use both the text and the images to answer the question."
            ),
        })
        for img in images:
            label = f"[Image {img.index + 1}"
            if img.page_number:
                label += f" — page {img.page_number}"
            label += f" — {img.width}×{img.height}px]"
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.base64_data,
                },
            })
    else:
        content.append({
            "type": "text",
            "text": f"Here is the relevant content from the document:\n\n{context}",
        })

    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def _call_claude(system: str, content: list[dict]) -> str:
    """Make a single Claude API call and return the text response."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    context: str,
    images: list[ExtractedImage] | None = None,
) -> str:
    """
    RAG answer — strictly grounded in document context and images only.
    Returns "I couldn't find that in the document." if not covered.
    """
    content = _build_content(question, context, images)
    return _call_claude(RAG_SYSTEM_PROMPT, content)


def answer_unrestricted(
    question: str,
    context: str,
    images: list[ExtractedImage] | None = None,
) -> str:
    """
    Unrestricted answer — given the same document context as the RAG answer,
    but free to expand beyond it using general knowledge.
    """
    content: list[dict] = []

    # Pass the document context so the answer is grounded but not restricted
    if context:
        content.append({
            "type": "text",
            "text": f"Here is relevant context from the document:\n\n{context}",
        })

    if images:
        content.append({
            "type": "text",
            "text": (
                f"The document also contains {len(images)} image(s) below. "
                "Use them to inform your answer if relevant."
            ),
        })
        for img in images:
            label = f"[Image {img.index + 1}"
            if img.page_number:
                label += f" — page {img.page_number}"
            label += f" — {img.width}×{img.height}px]"
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.base64_data,
                },
            })

    content.append({"type": "text", "text": f"Question: {question}"})
    return _call_claude(UNRESTRICTED_SYSTEM_PROMPT, content)


def answer_synthesized(
    question: str,
    rag_answer: str,
    unrestricted_answer: str,
) -> str:
    """
    Synthesized answer:
    - CASE 1: Document answers it → combine both sources
    - CASE 2: Related but not in document → use general knowledge
    - CASE 3: Completely unrelated → refuse politely
    """
    content = [{
        "type": "text",
        "text": (
            f"Question: {question}\n\n"
            f"RAG Answer (from the document):\n{rag_answer}\n\n"
            f"Unrestricted Answer (document context + general knowledge):\n{unrestricted_answer}\n\n"
            "Determine which CASE applies and respond accordingly."
        ),
    }]
    return _call_claude(SYNTHESIS_SYSTEM_PROMPT, content)