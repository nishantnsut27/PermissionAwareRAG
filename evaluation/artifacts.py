"""Verify frozen text artifacts across Git's LF/CRLF checkout conversion."""
import hashlib


def matches_sha256(content: bytes, expected: str) -> bool:
    """Accept only the original bytes or their LF/CRLF equivalent, not edits."""
    lf = content.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(candidate).hexdigest() == expected
               for candidate in (content, lf, lf.replace(b'\n', b'\r\n')))
