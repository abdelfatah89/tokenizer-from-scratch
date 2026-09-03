class BytesMaper:
    def __init__(self):
        self.id_to_bytes: dict[str, bytes] = {}
        self.bytes_to_id: dict[bytes, int] = {}
        self._build_byte_maps()

    def _build_byte_maps(self) -> None:
        """Build bidirectional maps between byte values (0-255) and their IDs."""
        for i in range(256):
            b = i.to_bytes()
            self.id_to_bytes[str(i)] = b
            self.bytes_to_id[b] = i

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        return [self.bytes_to_id[c.to_bytes()] for c in raw]

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.id_to_bytes[str(i)] for i in ids)
        return raw.decode("utf-8")
