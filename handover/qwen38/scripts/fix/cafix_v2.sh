#!/usr/bin/env bash
# Check (b): copy-ahead 2 (one set) with the row-padding writes on, off (MOE_STREAM_TEST_NO_ROW_PAD=1) and on again - the same text
# expected (the writes only replace bytes that meet zero activations). llama-completion, 1,795-token prompt, 200 slots pinned, 40 words.
R=/path/to/project; S=$R/bench/queue/qwen38/cafix_run.sh; L="$R/tools/model-lock.sh Qwen-fix"
export BS=0 PROMPT=$R/traces/qwen38/cafix/p1k9.txt
$L bash $S v2_pad 2 40 --moe-stream-slots 200
ENVS="MOE_STREAM_TEST_NO_ROW_PAD=1" $L bash $S v2_nopad 2 40 --moe-stream-slots 200
$L bash $S v2_pad_rep 2 40 --moe-stream-slots 200
cd $R/traces/qwen38/cafix && md5sum v2_pad.txt v2_nopad.txt v2_pad_rep.txt
grep -o "copy calls ([0-9]* of them zeros[^)]*)" v2_pad.log v2_nopad.log v2_pad_rep.log
