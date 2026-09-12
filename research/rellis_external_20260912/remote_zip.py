"""Read public ZIP members using validated HTTP byte ranges on Quest."""
import collections
import io
import re
import urllib.request


class RemoteZipReader(io.RawIOBase):
    def __init__(self, file_id, block_size=2**20):
        self.url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        self.position = 0
        self.block_size = block_size
        self.cache = collections.OrderedDict()
        self.transferred = 0
        req = urllib.request.Request(self.url, headers={"Range": "bytes=-65536"})
        with urllib.request.urlopen(req, timeout=60) as r:
            content_range = r.headers.get("Content-Range", "")
            m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if r.status != 206 or not m:
                raise ValueError("Server does not expose validated byte ranges")
            start, end, self.size = map(int, m.groups())
            self.tail_start = start
            self.tail = r.read(end - start + 2)
            if len(self.tail) != end - start + 1:
                raise ValueError("Truncated remote ZIP tail")
            self.transferred += len(self.tail)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError("Negative seek")
        self.position = pos
        return pos

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < stop:
            if self.position >= self.tail_start:
                part = self.tail[self.position - self.tail_start:stop - self.tail_start]
            else:
                index = self.position // self.block_size
                if index not in self.cache:
                    start = index * self.block_size
                    end = min(self.size, start + self.block_size) - 1
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                            raise ValueError("Mismatched HTTP range")
                        body = r.read(end - start + 2)
                    if len(body) != end - start + 1:
                        raise ValueError("Truncated HTTP range")
                    self.cache[index] = body
                    self.transferred += len(body)
                    if len(self.cache) > 64:
                        self.cache.popitem(last=False)
                chunk = self.cache[index]
                offset = self.position - index * self.block_size
                part = chunk[offset:offset + min(stop - self.position, len(chunk) - offset)]
            if not part:
                raise EOFError("Unexpected empty HTTP range")
            self.position += len(part)
            parts.append(part)
        return b"".join(parts)
