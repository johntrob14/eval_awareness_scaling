#!/bin/bash
# Background download of all exp02 checkpoints at pinned revisions (order = order of use).
set -u
dl() { echo "[$(date +%H:%M:%S)] start $1@$2"; /venv/main/bin/hf download "$1" --revision "$2" --quiet >/dev/null && echo "[$(date +%H:%M:%S)] done  $1" || echo "[$(date +%H:%M:%S)] FAILED $1"; }
dl Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca
dl Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17
dl Qwen/Qwen3-14B 40c069824f4251a91eefaf281ebe4c544efd3e18
dl Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654
dl Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
dl Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c
dl Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
dl Qwen/Qwen3.8-27B-FP8 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
dl Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc
dl Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
dl Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
echo "[$(date +%H:%M:%S)] all queued downloads finished"
