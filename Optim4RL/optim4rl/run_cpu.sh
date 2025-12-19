#!/bin/bash
# GPU 에러 회피용 CPU 실행 스크립트

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

python main.py "$@"
