"""Tiny helper for diagnostics that both print and save a text report."""

from pathlib import Path


class Report:
    """Accumulates report lines, echoing them to stdout as they are added."""

    def __init__(self, title: str | None = None, echo: bool = True):
        self.lines: list[str] = []
        self.echo = echo
        if title is not None:
            self.line("=" * 100)
            self.line(title)
            self.line("=" * 100)

    def line(self, text: str = ""):
        self.lines.append(text)
        if self.echo:
            print(text)

    def lines_from(self, texts):
        for text in texts:
            self.line(text)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def write(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text)
        return path
