# Contributing to TokenWing

## Help with anything

Code, ideas, models, hardware tests, docs: improve or add whatever you'd like. Your own ideas are just as welcome. If they happen to fit where we're heading, that's a nice bonus.

**Where we're heading:**

- **More models.**
- **More hardware:** macOS on Apple Silicon, Intel GPUs, and whatever you've got.
- **Faster replies with drafting** (MTP and n-gram).
- **Sharper predictions and smarter memory use.**

## How a change gets in

Bug fixes and small changes can go straight to a pull request. For a new feature or a big change, open a short issue first.

- One topic per pull request.
- New behaviour comes in as a setting, off by default.
- Fill in the pull request form: what changes, why, the impact on speed, memory and output, and how you tested it (your machine, and before and after numbers for speed). Write n/a where a box doesn't apply.
- Pull requests are squashed into one commit.
- You can use AI tools, as long as you understand every line you send.

Right now one person reviews everything, so a review can take a while and not everything can be accepted. Please don't take it personally.

By sending a pull request you agree that your work is under the [MIT licence](LICENSE).

## Testing your change

[Testing](docs/testing.md) has the details:

- `tokenwing-tools/testing/check.py` builds, runs the tests and tries a small model on your machine.
- `tokenwing-tools/testing/compare.py` measures before and after.

A change that shouldn't alter the output must still write the same text with the feature off. Small rounding differences between execution paths are fine.

[Architecture](docs/architecture.md) maps the code, and [Design decisions](docs/design-decisions.md) lists what has already been tried.

## Reporting a bug

Open an [issue](https://github.com/runmyai/tokenwing/issues/new/choose) and fill in the bug form. Not sure whether it's the app or the engine? Report it in [Run My AI](https://github.com/runmyai/run-my-ai/issues) and we'll move it.

For a question or an idea, use [Discussions](https://github.com/runmyai/tokenwing/discussions) or [Discord](https://discord.gg/BfpvUFp58).
