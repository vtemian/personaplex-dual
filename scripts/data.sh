#!/bin/sh
# scripts/data.sh: the 130 round-2 dialogues, the 16 TTS voices and the dv3a tape.
set -eu
: "${CHALKBOARD:?}" "${BOX:?}" "${TAPE:?}"
mkdir -p data/dialogues data/voices data/tape
cp "$CHALKBOARD"/data/derivative-r2/dlg-*.dsl data/dialogues/
cp -R "$CHALKBOARD"/data/derivative-r2/voices/. data/voices/
[ -f data/tape/speech.jsonl ] || scp "$BOX:$TAPE" data/tape/speech.jsonl
n_dsl=$(ls data/dialogues/*.dsl | wc -l | tr -d ' ')
n_tape=$(wc -l < data/tape/speech.jsonl | tr -d ' ')
n_voice=$(find data/voices -name '*.safetensors' | wc -l | tr -d ' ')
echo "dialogues=$n_dsl tape=$n_tape voices=$n_voice"
[ "$n_dsl" = 130 ] && [ "$n_tape" = 130 ] && [ "$n_voice" = 16 ] || { echo "data counts off"; exit 1; }
