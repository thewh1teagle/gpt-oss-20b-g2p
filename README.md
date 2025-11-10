# GPT-OSS 20B G2P

**[NOTE]** This is a work in progress.

This is a project to finetune the GPT-OSS 20B model for G2P.

## Setup

```console
uv sync
```

## Run

```console
uv run main.py
```

Based on [Unsloth's GPT-OSS 20B finetuning](https://unsloth.ai/blog/gpt-oss)


## TODO

- [ ] Add a CSV dataset
- [ ] Add WER/CER evaluation
- [ ] Evaluate with [Hebrew G2P Benchmark](https://github.com/thewh1teagle/heb-g2p-benchmark)