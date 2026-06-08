import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Book:
    title: str
    author: str = "Unknown"
    chapters: list["Chapter"] = field(default_factory=list)
    file_path: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(ch.content for ch in self.chapters)


@dataclass
class Chapter:
    title: str
    content: str


def parse_ebook(file_path: str) -> Book:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _parse_epub(path)
    elif suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".txt", ".text"):
        return _parse_txt(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _parse_epub(path: Path) -> Book:
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(f"EPUB support requires ebooklib and beautifulsoup4: {e}")

    book = epub.read_epub(str(path))
    title = _get_epub_title(book)
    author = _get_epub_author(book)
    chapters: list[Chapter] = []

    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
            elif lines and lines[-1] != "":
                lines.append("")

        if not lines:
            continue

        while lines and lines[-1] == "":
            lines.pop()

        content = "\n".join(lines)
        href = item.get_name()
        chapter_title = _extract_chapter_title(content) or href
        if chapter_title and content.startswith(chapter_title):
            content = content[len(chapter_title):].lstrip("\n")
        chapters.append(Chapter(title=chapter_title, content=content))

    if not chapters:
        chapters.append(Chapter(title="Full Text", content=title))

    return Book(title=title, author=author, chapters=chapters, file_path=str(path))


def _get_epub_title(book) -> str:
    titles = book.get_metadata("DC", "title")
    if titles:
        return str(titles[0][0])
    return "Unknown Title"


def _get_epub_author(book) -> str:
    authors = book.get_metadata("DC", "creator")
    if authors:
        return str(authors[0][0])
    return "Unknown"


def _extract_chapter_title(content: str) -> Optional[str]:
    match = re.match(
        r'^(?:Chapter\s+\d+|CHAPTER\s+\d+|[IVXLCDM]+\.?|[0-9]+\.?)\s*(.*)',
        content[:200],
    )
    if match:
        return match.group(0).strip()
    first_line = content.split("\n")[0].strip()
    if len(first_line) < 100 and not first_line.endswith("."):
        return first_line
    return None


def _parse_pdf(path: Path) -> Book:
    try:
        import fitz
    except ImportError:
        raise ImportError("PDF support requires PyMuPDF (fitz)")

    doc = fitz.open(str(path))
    title = path.stem
    parts: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            parts.append(text.strip())
    doc.close()
    return Book(
        title=title, author="",
        chapters=[Chapter(title="Full Text", content="\n\n".join(parts))],
        file_path=str(path),
    )


def _parse_txt(path: Path) -> Book:
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    chapters: list[Chapter] = []
    current: list[str] = []
    chapter_title = "Chapter 1"

    chapter_pattern = re.compile(
        r'^(?:Chapter\s+\d+|CHAPTER\s+\d+|[IVXLCDM]+\.?|[0-9]+\.)\s*.*$',
        re.IGNORECASE,
    )
    section_pattern = re.compile(
        r'^(?:Part|Section|Book|Volume)\s+\d+.*$', re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current.append("")
            continue
        if chapter_pattern.match(stripped) or section_pattern.match(stripped):
            if current:
                while current and current[-1] == "":
                    current.pop()
                chapters.append(Chapter(title=chapter_title, content="\n".join(current)))
            chapter_title = stripped
            current = []
        else:
            current.append(stripped)

    if current:
        while current and current[-1] == "":
            current.pop()
        chapters.append(Chapter(title=chapter_title, content="\n".join(current)))

    if not chapters:
        chapters = [Chapter(title="Full Text", content=content)]

    return Book(title=path.stem, author="", chapters=chapters, file_path=str(path))
