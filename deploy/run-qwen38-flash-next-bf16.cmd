@echo off
set "CONTAINER_NAME=morrigan-tokenwing-qwen38"
set "MODEL_DIR=E:\tokenwing\models\qwen38-flash-next-bf16"
set "MODEL_FILE=Qwen3.8-Flash-Next-BF16.tokenwing.gguf"
set "HOST_PORT=8084"
set "USE_CONTROL_SIDECARS=0"
call "%~dp0run-qwen38-flash-next.cmd"
