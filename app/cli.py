"""Command-line interface for customer-support response suggestions."""
import argparse
from pathlib import Path

from .core import load_model


def main(argv: list[str] | None = None) -> None:
    default_corpus = Path(__file__).resolve().parent.parent / "data" / "support_corpus.txt"
    parser = argparse.ArgumentParser(
        prog="support-response-generator",
        description="Suggest response continuations from anonymized support interactions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    suggest = sub.add_parser("suggest", help="suggest next words and a response continuation")
    suggest.add_argument("context", help="partial response written by the agent")
    suggest.add_argument("--corpus", default=str(default_corpus), help="path to the support corpus")
    suggest.add_argument("--n", type=int, choices=(2, 3), default=2, help="2 for bigram, 3 for trigram")
    suggest.add_argument("--k", type=float, default=0.5, help="Add-k smoothing value")
    suggest.add_argument("--words", type=int, default=12, help="maximum generated continuation length")
    suggest.add_argument("--top-k", type=int, default=3, help="number of next-word suggestions")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Support Response Generator is ready.")
        return
    if args.words < 1 or args.top_k < 1 or args.k <= 0:
        parser.error("--words, --top-k and --k must be greater than zero")

    model = load_model(args.corpus, n=args.n, k=args.k)
    suggestions = model.suggest_next(args.context, top_k=args.top_k)
    print("Next-word suggestions:")
    if suggestions:
        for word, probability in suggestions:
            print(f"  {word:<18} {probability:.3f}")
    else:
        print("  No suggestions found for this context.")
    continuation = model.generate(args.context, max_words=args.words)
    print("\nSuggested continuation:")
    print(f"  {continuation or '(no continuation available)'}")


if __name__ == "__main__":
    main()
