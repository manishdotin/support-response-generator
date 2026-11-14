"""Core text processing and n-gram language model for support responses."""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PII_PATTERNS = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "<email>"),
    (re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"), "<phone>"),
    (re.compile(r"\b(?:order|ticket|case|tracking)\s*(?:#|:|id)?\s*[A-Z0-9-]{4,}\b", re.I), "<reference>"),
    (re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I), "<url>"),
]

def anonymize_text(text: str) -> str:
    """Replace common direct identifiers and normalize whitespace."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()

def to_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]

def to_tokens(sentence: str) -> list[str]:
    return re.findall(r"<[a-z]+>|[a-z']+", sentence.lower())

def prepare_corpus(source: str | Path, destination: str | Path) -> list[str]:
    """Anonymize each non-empty source line and save the prepared corpus."""
    lines = [anonymize_text(line) for line in Path(source).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines

class NgramLM:
    """Count-based n-gram language model with Add-k smoothing."""
    def __init__(self, n: int = 2, k: float = 0.5):
        if n < 2:
            raise ValueError("n must be at least 2")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.n, self.k = n, k
        self.foll = defaultdict(Counter)
        self.V = {"</s>"}
        self.sents = []

    def fit(self, text: str | list[str]):
        self.sents = to_sentences(text) if isinstance(text, str) else list(text)
        self.foll = defaultdict(Counter)
        self.V = {"</s>"}
        for sentence in self.sents:
            tokens = ["<s>"] * (self.n - 1) + to_tokens(sentence) + ["</s>"]
            self.V.update(t for t in tokens if t != "<s>")
            for i in range(len(tokens) - self.n + 1):
                self.foll[tuple(tokens[i:i + self.n - 1])][tokens[i + self.n - 1]] += 1
        return self

    def probability(self, context: tuple[str, ...], word: str) -> float:
        counts = self.foll.get(context, Counter())
        total = sum(counts.values())
        return (counts[word] + self.k) / (total + self.k * len(self.V))

    def suggest_next(self, context: str, top_k: int = 3) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = to_tokens(context)
        ctx = tuple((["<s>"] * (self.n - 1) + tokens)[-(self.n - 1):])
        counts = self.foll.get(ctx, Counter())
        if not counts:
            return []
        total = sum(counts.values())
        ranked = [
            (word, (count + self.k) / (total + self.k * len(self.V)))
            for word, count in counts.items() if word != "</s>"
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_k]

    def generate(self, context: str = "", max_words: int = 12, seed: int = 0) -> str:
        if max_words < 1:
            return ""
        rng = random.Random(seed)
        prefix = to_tokens(context)
        generated = prefix[:]
        for _ in range(max_words):
            ctx = tuple((["<s>"] * (self.n - 1) + generated)[-(self.n - 1):])
            counts = self.foll.get(ctx, Counter())
            if not counts:
                break
            words, weights = zip(*[(w, c + self.k) for w, c in counts.items()])
            word = rng.choices(words, weights=weights, k=1)[0]
            if word == "</s>":
                break
            generated.append(word)
        return " ".join(generated[len(prefix):])

def load_model(corpus_path: str | Path, n: int = 2, k: float = 0.5) -> NgramLM:
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    return NgramLM(n=n, k=k).fit(path.read_text(encoding="utf-8"))

def milestone_1(source=None, destination=None):
    base = Path(__file__).resolve().parent.parent
    return prepare_corpus(source or base / "data/support_interactions_raw.txt",
                          destination or base / "data/support_corpus.txt")

def milestone_2(corpus_path=None, n=2, k=0.5):
    base = Path(__file__).resolve().parent.parent
    return load_model(corpus_path or base / "data/support_corpus.txt", n=n, k=k)

def milestone_3(model, context, max_words=12, top_k=3):
    return {"suggestions": model.suggest_next(context, top_k),
            "continuation": model.generate(context, max_words)}

def milestone_4(argv=None):
    from .cli import main
    return main(argv)
