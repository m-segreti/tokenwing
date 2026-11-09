# Notes for AI coding tools

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. The person sending the change is responsible for every line and must understand it.

- Build: see [docs/building.md](docs/building.md). After a library change, rebuild all targets.
- Test: `python3 tokenwing-tools/testing/check.py --model SMALL.gguf --build build`. See [docs/testing.md](docs/testing.md).
- New behaviour is a setting, off by default, and goes in [docs/settings.md](docs/settings.md) in the same change.
- With a feature off, the output must not change.
- System calls for reads, memory and thread placement go through the small named functions in `src/llama-moe-stream.cpp`.
- Don't write task labels, dates or private notes into code.
