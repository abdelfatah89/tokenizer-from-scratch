from typing import Dict, List


class BytesMapper:
    def __init__(self):
        self.id_to_bytes: Dict[int, bytes] = {}
        self.bytes_to_id: Dict[bytes, int] = {}
        self._build_byte_maps()

    def _build_byte_maps(self) -> None:
        """Build bidirectional maps between byte values
         (0-255) and their integer IDs."""
        for i in range(256):
            b = bytes([i])
            self.id_to_bytes[i] = b
            self.bytes_to_id[b] = i

    def encode(self, text: str) -> List[int]:
        """Convert text directly into a list of byte integers (0-255)."""
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from IDs safely without
        crashing on partial characters."""
        raw = b"".join(self.id_to_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")