# Convolutional SNN Classification Baselines

Multi-step convolutional spiking neural networks for **MNIST** and **CIFAR-10**, built with PyTorch and [SpikingJelly](https://github.com/fangwei123456/spikingjelly). This repository provides a compact baseline for static-image classification, with separate training, validation, and test entry points.

## Model and scope

- Ordinary learned convolutional and linear synapses, with `LIFNode` or `IFNode` spiking neurons.
- Static image features repeated over `T` timesteps; outputs are aggregated over time.
- Surrogate-gradient training with `ATan`, checkpointing, metric logs, and generated figures.

This is a **conventional Conv-SNN baseline**. It does not implement differentiable logic-gate synapses. It also does not include the CIFAR10-DVS experiments maintained in the separate SNN research workspace.

## Recorded experiments

The following historical runs are recorded in [the tracked result metadata](results/experiment_summary.json). They are individual-run observations, not multi-seed estimates or state-of-the-art claims.

| Dataset | Run | Epochs | T | Best validation accuracy | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| MNIST | `mnist_20260602_052040` | 100 | 4 | 99.43% | 99.36% |
| CIFAR-10 | `cifar10_20260602_072653` | 150 | 4 | 88.40% | 88.04% |
| CIFAR-10 | `cifar10_20260602_053054` | 100 | 4 | 87.76% | 86.92% |

Each training run selects its checkpoint using validation accuracy; `test.py` evaluates the official test split separately. The current summary script ranks runs by test accuracy before validation accuracy, so its generated “best run” label should not be interpreted as a validation-only comparison across configurations. No model-selection code has been changed in this documentation update.

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a short MNIST experiment:

```bash
python train.py \
  --dataset mnist \
  --data-root ./data \
  --output-dir ./runs/mnist \
  --epochs 5 \
  --batch-size 64 \
  --timesteps 4 \
  --device cpu
```

Evaluate its best validation checkpoint, replacing `<run>` with the generated run directory:

```bash
python test.py --run-dir ./runs/mnist/<run> --device cpu
```

The repository tracks result summaries and figures; the full historical run directories and checkpoints are not included. A fresh checkout requires training a model before using the evaluation command.

## Example training configurations

The following commands use the architecture sizes and training durations recorded above. Exact historical run metadata should be retained when reproducing or comparing results.

```bash
# MNIST, 100 epochs
python train.py --dataset mnist --data-root ./data \
  --output-dir ./runs/mnist --epochs 100 --batch-size 64 \
  --timesteps 4 --channels 64 --hidden-dim 256 --device cuda --amp

# CIFAR-10, 150 epochs
python train.py --dataset cifar10 --data-root ./data \
  --output-dir ./runs/cifar10 --epochs 150 --batch-size 64 \
  --timesteps 4 --channels 64 --hidden-dim 256 --device cuda --amp
```

## Experiment artifacts

| Artifact | Contents |
| --- | --- |
| `args.json` | Run configuration, including training and split seeds |
| `metrics.json` | Best-validation and final training metrics |
| `checkpoint_best.pt` | Checkpoint selected within a run |
| `test_metrics.json` | Test evaluation written by `test.py` |
| `train.log` | Per-epoch records |
| Generated SVG figures | Loss, validation accuracy, and confusion matrix |

Training uses a deterministic split of the official training data. The repository includes a [CPU smoke-test workflow](.github/workflows/smoke-test.yml); that workflow checks basic execution rather than reproducing the full experiments.

## Code and documentation

| Path | Purpose |
| --- | --- |
| `train.py` / `test.py` | Training and standalone evaluation |
| `snn_minimal/models/conv_snn.py` | Convolutional SNN architectures |
| `snn_minimal/data.py` / `engine.py` | Data preparation and training/evaluation loops |
| `scripts/summarize_runs.py` | Historical run aggregation and figure collection |
| [README.technical.md](README.technical.md) | Detailed technical documentation |
| [report.md](report.md) | Chinese experiment report |
| [REFERENCE_NOTES.md](REFERENCE_NOTES.md) | Upstream attribution and implementation boundaries |

Spiking neuron implementations, multi-step utilities, and reset helpers come from SpikingJelly. The repository implements the experiment pipeline, configuration handling, evaluation, and artifact packaging. `requirements.txt` references SpikingJelly commit `b049197b9589211418940c50031d3b7acb603bd6`.

## Limitations

The recorded results do not establish energy savings, hardware acceleration, or superiority over a matched ANN. Static-image repetition also does not test event-stream temporal modeling. Useful extensions include multiple seeds, controlled neuron/timestep ablations, matched ANN comparisons, and measured spike statistics.

## License

[MIT](LICENSE). Upstream dependencies retain their respective licenses.
