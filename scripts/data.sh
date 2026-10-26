#!/bin/sh
# scripts/data.sh: the 130 round-2 dialogues, the 16 TTS voices and the dv3a tape.
set -eu
: "${CHALKBOARD:?}" "${BOX:?}" "${TAPE:?}"
rm -rf data/dialogues data/voices
mkdir -p data/dialogues data/voices data/tape
cp "$CHALKBOARD"/data/derivative-r2/dlg-*.dsl data/dialogues/
cp -R "$CHALKBOARD"/data/derivative-r2/voices/. data/voices/
if [ ! -f data/tape/speech.jsonl ]; then
  scp "$BOX:$TAPE" data/tape/speech.jsonl.part
  mv data/tape/speech.jsonl.part data/tape/speech.jsonl
fi
n_dsl=$(ls data/dialogues/*.dsl | wc -l | tr -d ' ')
n_tape=$(wc -l < data/tape/speech.jsonl | tr -d ' ')
n_voice=$(find data/voices -name '*.safetensors' | wc -l | tr -d ' ')
echo "dialogues=$n_dsl tape=$n_tape voices=$n_voice"
[ "$n_dsl" = 130 ] && [ "$n_tape" = 130 ] && [ "$n_voice" = 16 ] || { echo "expected dialogues=130 tape=130 voices=16"; exit 1; }
