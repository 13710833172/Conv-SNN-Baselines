# Minimal Conv-SNN Baseline

A small, runnable Spiking Neural Network project for **MNIST** and **CIFAR-10**, built by referencing the `activation_based` examples and model definitions from [SpikingJelly](https://github.com/fangwei123456/spikingjelly) commit `b049197b9589211418940c50031d3b7acb603bd6`.

## Goals

- Keep the codebase runnable and easy to inspect
- Reuse the Conv-SNN training pattern from SpikingJelly
- Avoid copying the whole upstream framework into this repository

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (creates a timestamped run directory)
CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset mnist \
  --data-root ./data \
  --output-dir ./runs/mnist \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --device cuda \
  --amp

# 3. Evaluate best checkpoint on the test split
python test.py --run-dir ./runs/cifar10/cifar10_20260602_072653 --device cuda
```

## Project Layout

```
SNN/
├── train.py                 # Training entry point
├── test.py                  # Standalone test-set evaluation (best checkpoint)
├── requirements.txt
├── REFERENCE_NOTES.md       # Upstream reference mapping
├── data/                    # Downloaded datasets (gitignored)
├── runs/                    # Training outputs (gitignored)
└── snn_minimal/
    ├── data.py              # Datasets and DataLoaders
    ├── engine.py            # Epoch loop, checkpoints
    ├── plotting.py          # train.log, metrics.json, SVG curves
    └── models/
        └── conv_snn.py      # Conv-SNN model
```


| File                             | Role                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `train.py`                       | Train on the training split; each epoch runs validation; saves checkpoints and curves  |
| `test.py`                        | Load `checkpoint_best.pt` from a run directory and evaluate on the official test split |
| `snn_minimal/plotting.py`        | Logging and matplotlib SVG plots (invoked automatically by `train.py`)                 |
| `snn_minimal/engine.py`          | Shared `run_epoch`, checkpoint save/load                                               |
| `snn_minimal/data.py`            | MNIST / CIFAR-10 loaders and transforms                                                |
| `snn_minimal/models/conv_snn.py` | `ConvSNN` definition                                                                   |


## Train vs Validation vs Test


| Phase          | When                      | Data split                          | Script     |
| -------------- | ------------------------- | ----------------------------------- | ---------- |
| **Train**      | Every epoch               | Training set (`train=True`)         | `train.py` |
| **Validation** | Every epoch (after train) | Split from official training set    | `train.py` |
| **Test**       | After training (manual)   | Official test split (`train=False`) | `test.py`  |


During training, validation is created by splitting the official training set with a deterministic random split (`--val-ratio`, `--split-seed`). The official test split is reserved for `test.py` only.

There is **no** separate test phase inside `train.py`; use `test.py` when you need `test_metrics.json`.

## Run Directory

Each new training run creates a **timestamped** folder:

```text
<output-dir>/<dataset>_<YYYYMMDD_HHMMSS>/
```

Example: `runs/mnist/mnist_20260602_052040/`


| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `args.json`            | Full CLI arguments for this run                         |
| `checkpoint_latest.pt` | Latest epoch weights                                    |
| `checkpoint_best.pt`   | Best validation accuracy (used by `test.py` by default) |
| `train.log`            | One line per epoch (loss, acc, time, lr, …)             |
| `metrics.json`         | Compact summary: `final` and `best` epoch metrics       |
| `train_loss.svg`       | Training loss vs epoch (written when training finishes) |
| `val_acc.svg`          | Validation accuracy vs epoch                            |
| `confusion_matrix.svg` | Validation confusion matrix of `checkpoint_best.pt`     |
| `test_metrics.json`    | Created by `test.py` (not by `train.py`)                |


**Resume:** pass `--resume path/to/checkpoint_latest.pt`. Training continues in the **same** run directory; `train.log` is appended and curves cover the full history.

TensorBoard is **not** used (no `events.out.tfevents.`* files).

### `metrics.json` format

```json
{
  "final": {
    "epoch": 99,
    "train_loss": 1.83e-05,
    "val_acc": 0.9943
  },
  "best": {
    "epoch": 67,
    "train_loss": 3.67e-05,
    "val_acc": 0.9956
  }
}
```

Full per-epoch curves are in `train.log` and the SVG files; `metrics.json` only stores the last epoch and the best-validation epoch.

### `test_metrics.json` format

```json
{
  "checkpoint": ".../checkpoint_best.pt",
  "checkpoint_epoch": 67,
  "checkpoint_best_val_acc": 0.9956,
  "test_loss": 0.00085,
  "test_acc": 0.9956,
  "test_samples": 10000
}
```

## Latest Results Snapshot

The following numbers come from the latest completed run folders in this workspace.

### MNIST (`runs/mnist/mnist_20260602_052040`)

- `metrics.json`
  - best val acc: **0.9943** (epoch 58)
  - final val acc: **0.9933** (epoch 99)
- training setup (from `args.json` in the run): 100 epochs, `timesteps=4`, `channels=64`, `hidden_dim=256`, `neuron=lif`, `val_ratio=0.1`

### CIFAR-10 (`runs/cifar10/cifar10_20260602_072653`)

- `metrics.json`
  - best val acc: **0.8840** (epoch 141)
  - final val acc: **0.8746** (epoch 149)
- `test_metrics.json`
  - test acc: **0.8804**
  - test loss: **0.01859**
  - test samples: **10000**
- training setup (`args.json`): 150 epochs, `timesteps=4`, `channels=64`, `hidden_dim=256`, `neuron=lif`, `amp=true`, `val_ratio=0.1`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

- `torch`, `torchvision`
- `matplotlib` (training curves)
- `tqdm`
- `spikingjelly` (pinned git commit)

### GPU (CUDA)

Device selection in `train.py` / `test.py`:

1. `--device` if set (e.g. `cuda`, `cpu`, `mps`)
2. Else CUDA if `torch.cuda.is_available()`
3. Else MPS (Apple Silicon) if available
4. Else CPU

PyTorch must match your NVIDIA driver (e.g. CUDA 12.4 wheels on a 12.4 driver). Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Example: bind a specific GPU and enable mixed precision:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset cifar10 \
  --data-root ./data \
  --output-dir ./runs/cifar10 \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --device cuda \
  --amp
```

`--amp` only takes effect on CUDA.

## Training

### MNIST

```bash
python train.py \
  --dataset mnist \
  --data-root ./data \
  --output-dir ./runs/mnist \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --device cuda \
  --amp
```

### CIFAR-10

```bash
python train.py \
  --dataset cifar10 \
  --data-root ./data \
  --output-dir ./runs/cifar10 \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --channels 64 \
  --hidden-dim 256 \
  --device cuda \
  --amp
```

### `train.py` arguments


| Argument                | Default    | Description                                           |
| ----------------------- | ---------- | ----------------------------------------------------- |
| `--dataset`             | (required) | `mnist` or `cifar10`                                  |
| `--data-root`           | `./data`   | Dataset download root                                 |
| `--output-dir`          | `./runs`   | Parent directory for timestamped runs                 |
| `--epochs`              | `5`        | Number of epochs                                      |
| `--batch-size`          | `64`       | Batch size                                            |
| `--num-workers`         | `2`        | DataLoader workers                                    |
| `--timesteps`           | `4`        | SNN time steps `T`                                    |
| `--channels`            | `64`       | Base conv channel width                               |
| `--hidden-dim`          | `256`      | Classifier hidden size                                |
| `--dropout`             | `0.25`     | Dropout rate                                          |
| `--neuron`              | `lif`      | `lif` or `if`                                         |
| `--tau`                 | `2.0`      | LIF time constant                                     |
| `--optimizer`           | `adam`     | `adam` or `sgd`                                       |
| `--lr`                  | `1e-3`     | Learning rate                                         |
| `--momentum`            | `0.9`      | SGD momentum                                          |
| `--weight-decay`        | `0.0`      | Weight decay                                          |
| `--device`              | auto       | `cuda`, `cpu`, `mps`, or omit for auto                |
| `--seed`                | `42`       | Random seed                                           |
| `--amp`                 | off        | Mixed precision (CUDA only)                           |
| `--resume`              | none       | Checkpoint path; continues in same run dir            |
| `--val-ratio`           | `0.1`      | Fraction of official training set used for validation |
| `--split-seed`          | `42`       | Random seed for deterministic train/val split         |
| `--limit-train-batches` | none       | Cap train batches (smoke test)                        |
| `--limit-val-batches`   | none       | Cap val batches (smoke test)                          |


### Training loop (per epoch)

1. Train one epoch on the training split
2. Step `CosineAnnealingLR`
3. Validate on the held-out validation split from the official training set
4. Save `checkpoint_latest.pt`; update `checkpoint_best.pt` if val acc improves
5. Append `train.log` and update `metrics.json` via `plotting.py`
6. After all epochs: write `train_loss.svg`, `val_acc.svg`, and `confusion_matrix.svg`

## Testing

After training, evaluate the best checkpoint:

```bash
python test.py --run-dir ./runs/mnist/mnist_20260601_105849 --device cuda
```

- Reads `args.json` from the run directory to rebuild the model and data settings  
- Default checkpoint: `<run-dir>/checkpoint_best.pt`  
- Writes `<run-dir>/test_metrics.json`  
- Uses the full test set unless `--limit-test-batches` is set

### `test.py` arguments


| Argument               | Default              | Description                      |
| ---------------------- | -------------------- | -------------------------------- |
| `--run-dir`            | (required)           | Path to a training run directory |
| `--checkpoint`         | `checkpoint_best.pt` | Override checkpoint path         |
| `--data-root`          | from `args.json`     | Override data root               |
| `--batch-size`         | from `args.json`     | Override batch size              |
| `--num-workers`        | from `args.json`     | Override workers                 |
| `--device`             | auto                 | Same as training                 |
| `--limit-test-batches` | none                 | Cap batches (smoke test only)    |


## Smoke Test

Verify install and data paths with a tiny run:

```bash
python train.py \
  --dataset mnist \
  --epochs 1 \
  --batch-size 8 \
  --timesteps 2 \
  --limit-train-batches 2 \
  --limit-val-batches 2

python test.py \
  --run-dir ./runs/mnist/mnist_<timestamp> \
  --limit-test-batches 2
```

Replace `mnist_<timestamp>` with the directory printed at the end of training.

---

## Detailed Network Structure

The model is defined in `snn_minimal/models/conv_snn.py`.

### Common Design

- Multi-step SNN using SpikingJelly `activation_based` modules  
- Static images repeated across `T` time steps (no Poisson encoding)  
- Input shape: MNIST `[N, 1, 28, 28]`, CIFAR-10 `[N, 3, 32, 32]`  
- After expansion: `[T, N, C, H, W]`  
- Neurons: `LIFNode` (configurable `tau`) or `IFNode`, surrogate `ATan`  
- Readout: mean firing rate over time → `[N, 10]`

```text
output = mean_t(spike_logits[t]), t = 1 ... T
```

### SNN Principle and Spike Equations

The current network is a feed-forward convolutional SNN. Its temporal behavior does not come from video or event streams; it comes from the **internal membrane state** of the spiking neurons.

In this repository, one static image is copied to all `T` time steps:

```text
x[1] = x[2] = ... = x[T] = input_image
```

Even though the input frame is the same at every step, the neuron state evolves over time because each spiking neuron keeps an internal membrane potential and performs threshold-and-reset dynamics.

We use the following notation:

- `x[t]`: input current or input feature at time step `t`
- `h[t]`: membrane potential before spike generation at time step `t`
- `s[t]`: output spike at time step `t`
- `v[t]`: membrane potential after spike/reset at time step `t`
- `V_th`: firing threshold
- `V_reset`: reset voltage

For the current code, the experiments use `--neuron lif`, so the actual runs are based on **LIFNode**. The code can also switch to **IFNode** by passing `--neuron if`.

#### 1. LIF neuron used in the current experiments

By default, `ConvSNN` constructs:

```text
neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
```

with the default SpikingJelly hard-reset behavior. Conceptually, the forward dynamics are:

```text
h[t] = v[t-1] + (x[t] - (v[t-1] - V_reset)) / tau
s[t] = Theta(h[t] - V_th)
v[t] = V_reset, if s[t] = 1
v[t] = h[t],     if s[t] = 0
```

In the default configuration of SpikingJelly:

- `tau = 2.0` in this project unless changed by `--tau`
- `V_th = 1.0`
- `V_reset = 0.0`

So with the default settings, the charging equation can be viewed as:

```text
h[t] = v[t-1] + (x[t] - v[t-1]) / 2
```

Interpretation:

- the neuron integrates incoming input `x[t]`
- at the same time, its membrane leaks toward `V_reset`
- once the membrane crosses threshold, it emits a spike
- after emitting the spike, the membrane is reset

This is why LIF is called **Leaky Integrate-and-Fire**.

#### 2. Optional IF neuron

If you run with `--neuron if`, the neuron becomes an ideal integrator without leakage:

```text
h[t] = v[t-1] + x[t]
s[t] = Theta(h[t] - V_th)
v[t] = V_reset, if s[t] = 1
v[t] = h[t],     if s[t] = 0
```

Compared with LIF:

- IF keeps accumulating input without decay
- LIF adds a leak term, which usually gives more stable temporal dynamics

#### 3. Spike generation formula

The spike itself is binary and is produced by the threshold function:

```text
s[t] = Theta(h[t] - V_th)
```

where:

```text
Theta(z) = 1, if z >= 0
Theta(z) = 0, if z < 0
```

So a spike means:

- membrane potential reached threshold
- the neuron "fires" at that time step

No spike means:

- the neuron keeps integrating information
- the membrane state is preserved for the next time step

#### 4. Why surrogate gradient is needed

The true threshold function `Theta(.)` is not suitable for ordinary backpropagation because its derivative is zero almost everywhere and undefined at the threshold.

So during training, the code uses:

```text
surrogate.ATan()
```

This means:

- **forward pass** still uses a binary spike
- **backward pass** replaces the derivative of the hard threshold with a smooth approximation centered near the threshold

Intuitively, this allows gradients to flow through the spike function and makes end-to-end training possible.

#### 5. Temporal readout in this model

The final classifier layer is also a spiking layer, so the network output is not treated as ordinary ANN logits. Instead, the model uses the average firing activity over time:

```text
y = (1 / T) * sum_t s_out[t]
```

where `y` has shape `[N, 10]`.

This means the network predicts classes using **firing rate**:

- a larger value in class `k` means the output neuron for class `k` fired more often over the `T` steps
- prediction is made by `argmax(y)`

#### 6. Why state reset is necessary

Because spiking neurons carry membrane state across time, each batch must be reset after forward/backward:

```text
functional.reset_net(model)
```

Otherwise:

- the membrane from batch `n` would leak into batch `n+1`
- training and evaluation would become incorrect

So the temporal recurrence exists **within one sample batch across time steps**, but must not continue across different mini-batches.

### Spiking Block Pattern

```text
Conv2d -> BatchNorm2d -> SpikingNeuron -> Pooling
```

Classifier:

```text
Flatten -> Dropout -> Linear -> SpikingNeuron -> Dropout -> Linear -> SpikingNeuron
```

### MNIST Architecture

Defaults: `timesteps=4`, `channels=64`, `hidden_dim=256`, `neuron=lif`


| Stage          | Layer                                                                             | Output Shape                  |
| -------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| Input          | Static image                                                                      | `[N, 1, 28, 28]`              |
| Time expansion | Repeat `T` times                                                                  | `[T, N, 1, 28, 28]`           |
| Block 1        | `Conv2d(1→64, k=3, pad=1)` + BN + LIF/IF + `MaxPool2d(2)`                         | `[T, N, 64, 14, 14]`          |
| Block 2        | `Conv2d(64→128, k=3, pad=1)` + BN + LIF/IF + `MaxPool2d(2)`                       | `[T, N, 128, 7, 7]`           |
| Classifier     | Flatten → Dropout → Linear(6272→256) → LIF/IF → Dropout → Linear(256→10) → LIF/IF | `[N, 10]` after mean over `T` |


`6272 = 128 × 7 × 7`. If `channels` changes, the second block uses `2 × channels`.

### CIFAR-10 Architecture

Defaults: same as MNIST; four conv blocks before classifier.


| Stage      | Output Shape (after block)                                  |
| ---------- | ----------------------------------------------------------- |
| Blocks 1–2 | `[T, N, 64, 32, 32]` then `[T, N, 64, 16, 16]` (with pool)  |
| Blocks 3–4 | `[T, N, 128, 16, 16]` then `[T, N, 128, 8, 8]` (with pool)  |
| Classifier | `Linear(8192→256)` → `Linear(256→10)`, `8192 = 128 × 8 × 8` |


### Multi-Step Forward

1. Expand `[N, C, H, W]` → `[T, N, C, H, W]`
2. Forward through encoder + classifier
3. Mean over time → `[N, 10]`

## Loss, Optimizer, and SNN State Reset

**Loss:** MSE between firing-rate output and one-hot labels (not cross-entropy on logits).

```text
loss = MSE(output_firing_rate, one_hot_label)
pred = argmax(output_firing_rate, dim=1)
```

**Optimizer:** `Adam` or `SGD`; scheduler `CosineAnnealingLR` (stepped once per epoch).

**State reset:** after every batch, `functional.reset_net(model)` clears membrane state so batches do not leak temporal information.

## Data Processing

From `snn_minimal/data.py`:

- **MNIST:** `ToTensor` + `Normalize((0.1307,), (0.3081,))`  
- **CIFAR-10 train:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, normalize  
- **CIFAR-10 test:** `ToTensor` + normalize only

DataLoaders: train `shuffle=True`, `drop_last=True`; test/val `shuffle=False`; `pin_memory=True` on CUDA.

Current split policy:

- `train.py`: official training set is split into train/val (default `90% / 10%`)
- `test.py`: official test set is used only for final evaluation

## Checkpoint Contents

Each `.pt` file stores:

- `model`, `optimizer`, `scheduler` (if present)  
- `epoch`, `best_acc`, `args`

## Simplifications (by design)

- No Poisson encoder, voting layer, or residual blocks  
- No distributed training or event-based datasets  
- Focus: correct, readable baseline for remote GPU runs

