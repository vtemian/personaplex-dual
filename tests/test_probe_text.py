from dual.probe_text import (
    ANSWER,
    ECHO,
    IGNORE,
    classify,
    content_ids,
    echo_fraction,
    words_of,
)

# A fake SPM: id 4 + i is the word VOCAB[i]; specials 0..3 decode to nothing.
VOCAB = ["so", "what", "is", "the", "slope", "here?", "let's", "see,", "two", "x"]


def decode(ids):
    return " ".join(VOCAB[i - 4] for i in ids if i >= 4)


def ids(*words):
    return [4 + VOCAB.index(w) for w in words]


def test_words_of_lowercases_and_strips_punctuation():
    assert words_of("So, what IS the slope here?") == [
        "so",
        "what",
        "is",
        "the",
        "slope",
        "here",
    ]
    assert words_of("  -- ... ") == []
    assert words_of("let's") == ["lets"]


def test_content_ids_drops_the_four_specials():
    assert content_ids([3, 0, 7, 3, 1, 9, 2, 3]) == [7, 9]


def test_echo_fraction_counts_words_in_order():
    typed = ["what", "is", "the", "slope"]
    assert echo_fraction(typed, ["what", "is", "the", "slope"]) == 1.0
    # in order, with words between
    assert echo_fraction(typed, ["so", "what", "x", "is", "two"]) == 0.5
    # out of order: only "slope" then nothing after it
    assert echo_fraction(typed, ["slope", "the", "is", "what"]) == 0.25
    assert echo_fraction([], ["what"]) == 0.0


def test_ignore_when_fewer_than_four_content_tokens():
    emitted = [3, 3, 0, *ids("let's", "see,", "two"), 2, 3]
    assert classify(["what", "is"], emitted, decode) == IGNORE


def test_echo_when_half_the_typed_words_come_back_in_order():
    typed = ["what", "is", "the", "slope", "here?"]
    emitted = [3, *ids("so", "what", "is", "the"), 3, *ids("x", "two")]
    assert classify(typed, emitted, decode) == ECHO


def test_answer_otherwise():
    typed = ["what", "is", "the", "slope", "here?"]
    emitted = [3, *ids("let's", "see,", "two", "x", "so")]
    assert classify(typed, emitted, decode) == ANSWER


def test_echo_reads_only_the_first_forty_content_tokens():
    typed = ["what", "is", "the", "slope"]
    late = [*ids("two") * 40, *ids("what", "is", "the", "slope")]
    assert classify(typed, late, decode) == ANSWER


def test_a_typed_word_moshi_skipped_does_not_hide_the_ones_after_it():
    assert echo_fraction(["foo", "what", "is"], ["what", "is"]) == 2 / 3
