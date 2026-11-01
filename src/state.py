"""Input identity helpers shared by UI and stale-state regression tests."""
import hashlib
import json
from collections.abc import MutableMapping


def fingerprint(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        value = part if isinstance(part, bytes) else json.dumps(part, sort_keys=True, default=str).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def sync_inputs(state: MutableMapping, key: str, identity: str, results: tuple[str, ...]) -> bool:
    """Invalidate results before rendering when any relevant input changes."""
    changed = state.get(key) != identity
    if changed:
        for result in results:
            state.pop(result, None)
        state[key] = identity
    return changed
