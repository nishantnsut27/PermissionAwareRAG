"""Bounded conversation history.

History is untrusted user data. It is used for exactly one purpose: resolving
references in the current question ("which one has higher margin?"). It never
contributes to identity, scope or any authorization decision — the retrieval
filter is derived solely from the server-side identity, so history can bias
what is *searched for* but never what is *searchable*.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config

VALID_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


def parse_history(raw, cfg: config.AnsweringConfig | None = None,
                  *, user_only: bool = False) -> list[Turn]:
    """Validate and bound caller-supplied history. Unknown roles are dropped."""
    cfg = cfg or config.DEFAULT_ANSWERING
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("conversation must be a list of {role, content} objects")
    if len(raw) > cfg.max_history_turns * 4:
        raise ValueError("conversation contains too many turns")

    turns: list[Turn] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("conversation entries must be objects")
        if set(entry) != {"role", "content"}:
            raise ValueError("conversation entries must contain only role and content")
        role = str(entry.get("role", "")).strip().lower()
        content = entry.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            raise ValueError("conversation roles/content are invalid")
        if len(content) > cfg.max_history_chars * 4:
            raise ValueError("conversation entry exceeds the configured limit")
        content = content.strip()
        if content:
            turns.append(Turn(role, content[:cfg.max_history_chars]))

    if user_only:
        turns = [turn for turn in turns if turn.role == "user"]
    turns = turns[-cfg.max_history_turns:] if cfg.max_history_turns else []
    budget = cfg.max_history_chars
    kept: list[Turn] = []
    for turn in reversed(turns):
        if budget <= 0:
            break
        kept.append(Turn(turn.role, turn.content[:budget]))
        budget -= len(turn.content)
    kept.reverse()
    return kept


def render(turns: list[Turn]) -> str:
    return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in turns)


def retrieval_query(turns: list[Turn], question: str,
                    cfg: config.AnsweringConfig | None = None) -> str:
    """Query text used for embedding and lexical search.

    Prior user turns are prepended so follow-up questions keep their referents.
    This only affects ranking; the authorization filter is unaffected.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    previous = [t.content for t in turns if t.role == "user"][-2:]
    if not previous:
        return question
    combined = " ".join([*previous, question])
    return combined[-(cfg.max_query_chars + cfg.max_history_chars):]
