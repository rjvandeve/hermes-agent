"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record and timeline display, not
# provider-visible message content. The request builder strips them from every
# outgoing copy and the token estimator ignores them: one set, so an estimate
# never prices bytes the provider never receives (an edit's inline_diff in
# display_metadata is ~9KB and would trigger premature compaction).
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id"})

# Hermes-private truncation-recovery markers. History keeps them (retry de-dupe,
# compressor / crash-persisted synthetic-turn checks); no provider copy or estimate does.
PRIVATE_RECOVERY_MESSAGE_FIELDS = frozenset({
    "_length_continuation_fragment", "_length_continuation_nudge", "_tool_call_truncation_nudge",
})

# The one set the request builder strips from every outgoing copy and the estimator skips.
NON_WIRE_MESSAGE_FIELDS = PERSISTENCE_ONLY_MESSAGE_FIELDS | PRIVATE_RECOVERY_MESSAGE_FIELDS

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message
