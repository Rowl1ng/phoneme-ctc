#!/usr/bin/env bash
#documentsDir=/home/rowl1ng/PycharmProjects/speech/little_prince_1/Audio
#mkdir -p ${documentsDir}/test
#FILES=(${documentsDir}/*wav)
#for f in "${FILES[@]}"; do
#	fname=`basename $f .wav`
#	sox -v 0.8 "$f"  ${documentsDir}/test/${fname}.wav channels 1 rate -L -s 16000
#done
#/home/rowl1ng/PycharmProjects/speech/little_prince_1/Audio/test/boy_huang_01.wav
python phoneme_ctc.py decode -m model -f