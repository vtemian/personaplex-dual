"""Every number the three streams agree on.

All measured, none guessed; the source of each is named so a later reader can
re-measure it.
"""

FRAME_RATE = 12.5
CODEBOOKS = 8
STREAMS = 1 + 2 * CODEBOOKS  # text, moshi's 8, the pupil's 8
CONTEXT = 3000  # moshi's temporal context, frames

# moshi's acoustic delays, moshi/models/loaders.py:118 (0.2.13): text, own 8, pupil 8.
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)

# The four reserved SPM ids (chalkboard/grid.py:11-31).
CUE = 0  # <unk>: the frame before a later word
TURN = 1  # <s>: opens a teacher turn
END = 2  # </s>: after the turn's last piece
PAD = 3  # <pad>: silence on the text stream
TEXT_CARD = 32000
AUDIO_CARD = 2048

# PersonaPlex (NVIDIA/personaplex moshi/models/lm.py): moshi's own stream while it
# reads a prompt, and the pupil stream while text is typed. Never swap them.
SILENCE = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE = (430, 1268, 381, 1611, 1095, 1495, 56, 472)

# chalkboard training/moshi/stream.py:35, 2 s of silence through Mimi: the pupil stream
# after a spoken turn, which is real silence and not the typed marker.
MIMI_SILENCE = (1049, 1700, 1626, 546, 306, 1443, 1871, 2008)

# PersonaPlex's server: int(0.5 * 12.5) frames of pad after a text prompt.
GAP_FRAMES = 6

# Frames kept ahead of a turn's first word, chalkboard training/moshi/cut_turns.py:33.
LEAD = 4

IGNORE = -100

PERSONAPLEX = "personaplex"
MOSHIKO = "moshiko"
BASES = (PERSONAPLEX, MOSHIKO)
PERSONAPLEX_FILES = {
    "moshi": "model.safetensors",
    "mimi": "tokenizer-e351c8d8-checkpoint125.safetensors",
    "spm": "tokenizer_spm_32k_3.model",
}
MOSHIKO_REPO = "kyutai/moshiko-pytorch-bf16"
