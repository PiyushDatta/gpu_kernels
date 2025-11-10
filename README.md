# GPU Kernels

# Setup

1. `uv venv`
2. `source /.venv/bin/activate`
3. `uv pip install -e .`

# Examples

`./popcorn-cli --help`

`./popcorn-cli submit --gpu B200 --leaderboard grayscale_v2 --mode leaderboard submission.py`

```
leaderboard submit \
  --op add \
  --overload Tensor \
  --dsl cutedsl \
  --device A100 \
  --file add_implementation_v1.py
```

```
python ./server/main.py
leaderboard list
```
