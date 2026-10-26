"""The probe's verdict on what moshi said after a typed turn. Pure: no torch.

`answer`, `echo` or `ignore`, read off the text tokens moshi emitted in its free run.
Only word pieces count: the four reserved SPM ids (`<unk> <s> </s> <pad>`) carry no
words, and moshi's text stream is mostly `<pad>` and the `<unk>` cue before a word.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

ANSWER = "answer"
ECHO = "echo"
IGNORE = "ignore"
CLASSES = (ANSWER, ECHO, IGNORE)

FIRST_PIECES = 40  # the window an echo is looked for in
ECHO_SHARE = 0.5  # of the typed words, in order
MIN_PIECES = 4  # fewer word pieces than this in the whole free run is `ignore`
SPECIALS = 4  # ids 0..3

Decode = Callable[[list[int]], str]


def words_of(text: str) -> list[str]:
    out = ["".join(c for c in word.lower() if c.isalnum()) for word in text.split()]
    return [w for w in out if w]


def content_ids(ids: Sequence[int]) -> list[int]:
    return [int(i) for i in ids if int(i) >= SPECIALS]


def echo_fraction(typed: Sequence[str], emitted: list[str]) -> float:
    """Share of the typed words found in `emitted` in order (a greedy subsequence)."""
    if not typed:
        return 0.0
    found = 0
    at = 0
    for word in typed:
        try:
            at = emitted.index(word, at) + 1
        except ValueError:
            continue  # moshi skipped it; the next typed word may still follow
        found += 1
    return found / len(typed)


def classify(
    typed_words: Sequence[str], emitted_ids: Sequence[int], decode: Decode
) -> str:
    pieces = content_ids(emitted_ids)
    if len(pieces) < MIN_PIECES:
        return IGNORE
    typed = words_of(" ".join(typed_words))
    emitted = words_of(decode(pieces[:FIRST_PIECES]))
    return ECHO if echo_fraction(typed, emitted) >= ECHO_SHARE else ANSWER
