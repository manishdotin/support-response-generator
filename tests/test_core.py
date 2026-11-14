from app.core import NgramLM, anonymize_text, prepare_corpus
from app.cli import main


def test_anonymizes_common_identifiers():
    result = anonymize_text("Email alex@example.com or call 555-123-4567 about order #AB12345")
    assert "<email>" in result
    assert "<phone>" in result
    assert "<reference>" in result
    assert "alex@example.com" not in result


def test_prepare_corpus_writes_clean_lines(tmp_path):
    source, destination = tmp_path / "raw.txt", tmp_path / "clean.txt"
    source.write_text("Contact me at alex@example.com\n\nThanks for your help.")
    result = prepare_corpus(source, destination)
    assert len(result) == 2
    assert "<email>" in destination.read_text()


def test_ngram_smoothing_and_suggestions():
    model = NgramLM(n=2, k=0.5).fit("thank you for your order. thank you for your help.")
    suggestions = model.suggest_next("thank you", top_k=2)
    assert suggestions
    assert suggestions[0][0] == "for"
    assert model.probability(("thank",), "unseen") > 0


def test_trigram_and_generation():
    model = NgramLM(n=3).fit("please share your order number. please share your email address.")
    assert model.suggest_next("please share your")
    assert isinstance(model.generate("please share your", max_words=4), str)


def test_cli_suggest(capsys):
    main(["suggest", "thank you for contacting", "--words", "5", "--top-k", "2"])
    output = capsys.readouterr().out
    assert "Next-word suggestions:" in output
    assert "Suggested continuation:" in output


def test_cli_hello(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out
