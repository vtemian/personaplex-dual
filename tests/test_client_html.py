import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "dual" / "client.html"


def test_page_exists_and_speaks_the_wire():
    text = PAGE.read_text()
    assert text.lstrip().lower().startswith("<!doctype html>")
    # the kinds, as dual.engine defines them
    assert re.search(r"READY\s*=\s*0", text)
    assert re.search(r"AUDIO\s*=\s*1", text)
    assert re.search(r"TEXT\s*=\s*2", text)
    assert re.search(r"TYPED\s*=\s*3", text)
    assert re.search(r"STATUS\s*=\s*4", text)
    # 24 kHz int16, 1920-sample frames, the /ws path
    assert "24000" in text and "1920" in text and "32767" in text
    assert "/ws" in text
    assert "AudioWorklet" in text or "audioWorklet" in text
    assert 'id="text"' in text  # the typed-turn box
