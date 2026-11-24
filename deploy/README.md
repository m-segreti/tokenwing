# Local Qwen3.8 Flash Next deployment

This deployment runs the TokenWing-packaged Qwen3.8 Flash Next model as one
standalone Docker container on the local RTX 3080 Ti through TokenWing's
OpenAI-compatible server.

- Host endpoint: `http://127.0.0.1:8084`
- Configured context: 131,072 tokens (the model supports 262,144)
- Model storage: `E:\tokenwing\models\qwen38-flash-next`
- Durable conversation-state cache: `E:\tokenwing\cache\qwen38-flash-next`
- Container memory ceiling: 46 GiB with swap disabled
- CUDA build: CUDA 13, compute capability 8.6

The model directory is writable because TokenWing creates its measured expert
usage sidecar next to the model. The downloaded weights themselves are not
rewritten.

Start or recreate the container from this directory with:

```bat
run-qwen38-flash-next.cmd
```

The launcher removes only an existing `morrigan-tokenwing-qwen38` container,
then recreates it with the fixed model mount, port, memory ceiling, restart
policy, health check, 131,072-token context configuration, and a 16 GiB
disk-backed prompt/KV cache. Completed replies are persisted so a displaced
single live slot can restore the longest matching conversation prefix instead
of replaying the entire checkpoint.

Useful checks:

```sh
curl http://127.0.0.1:8084/health
curl http://127.0.0.1:8084/v1/models
curl http://127.0.0.1:8084/moe-stream/stats
```
