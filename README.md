# TFCLIP

TFCLIP is a reproduction workspace for **frozen-feature few-shot CLIP adaptation baselines**. The goal is to evaluate a set of training-free, closed-form, shallow, metric, cache-based, Gaussian, and kernel-based CLIP adaptation methods under one unified protocol.

This repository is intended to support the ClipGame paper experiments, where ClipGame is compared against methods that keep a CLIP-family vision-language backbone frozen and adapt only the decision rule or lightweight evidence computed from support-set features.

## Scope

All reproduced methods should use the same experimental conditions whenever possible:

- Same CLIP-family backbone and preprocessing.
- Same class names and prompt templates.
- Same few-shot support splits.
- Same random seeds.
- Same train/validation/test protocol.
- No test-set visibility during adaptation.
- No image encoder or text encoder fine-tuning.

All model variants should be routed through `open_clip_torch`. This keeps
OpenAI CLIP, OpenCLIP checkpoints, SigLIP-style checkpoints, and different ViT
sizes behind the same image/text feature interface.

Prompt-learning and neural-adapter methods such as CoOp, CoCoOp, MaPLe, PromptSRC, and CLIP-Adapter are useful related work, but they are not the primary baselines in this project because they optimize task-specific neural parameters.

## Baselines

| Method | Category | Code source | Reproduction plan |
|---|---|---|---|
| Zero-shot CLIP | Prompting | [openai/CLIP](https://github.com/openai/CLIP) | Implement directly in this repo |
| Prompt ensemble | Prompting | [openai/CLIP](https://github.com/openai/CLIP) | Implement directly in this repo |
| Linear probe | Shallow classifier | [openai/CLIP](https://github.com/openai/CLIP), [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Implement directly in this repo |
| Ridge classifier | Shallow classifier | [scikit-learn RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) | Implement directly in this repo |
| SVM | Shallow classifier | [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html) | Implement directly in this repo |
| Nearest centroid | Metric classifier | [scikit-learn NearestCentroid](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) | Implement directly in this repo |
| kNN / soft kNN | Metric classifier | [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) | Implement kNN directly; add cosine-weighted soft kNN |
| Tip-Adapter | Cache-based adaptation | [gaopengcuhk/Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) | Port or wrap official implementation |
| APE | Adaptive prior refinement | [yangyangyang127/APE](https://github.com/yangyangyang127/APE) | Port or wrap official implementation |
| LP++ | Text-informed linear probe | [FereshteShakeri/FewShot-CLIP-Strong-Baseline](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline) | Port or wrap official implementation |
| GDA-CLIP | Gaussian discriminant analysis | [mrflogs/ICLR24](https://github.com/mrflogs/ICLR24) | Port or wrap official implementation |
| ProKeR | Kernel-based adaptation | [ybendou/ProKeR](https://github.com/ybendou/ProKeR) | Port or wrap official implementation |

The local runner exposes `knn` and `soft_knn` as separate method names, but they
belong to the same kNN baseline family in the paper table.

Official-style method defaults are recorded in
`configs/methods/official_feature_space.yaml`, and current calibration notes are
kept in `docs/calibration.md`.

## Current Runner

The first unified runner is feature-space based: it extracts frozen image
features and text classifier features from an `open_clip` model, then feeds the
same cached train/val/test features to every baseline.
The main image feature cache is keyed by dataset, model, and split seed, so
datasets with seed-dependent train/val splits do not accidentally share cached
features across seeds.
Methods whose official code uses support-set augmentation or support-only train
transforms additionally create a raw support feature cache keyed by dataset,
model, seed, shots, `augment_epoch`, transform policy, and raw feature space.
Tip-Adapter and APE average raw augmented support views before normalizing;
LP++, GDA-CLIP, and ProKeR normalize each support view before fitting. APE uses
its official template-plus-CuPL text prompt cache.

```bash
python3 scripts/audit_data.py
python3 scripts/audit_method_calibration.py

python3 scripts/extract_features.py \
  --datasets dtd food101 oxford_pets flowers102 fgvc_aircraft eurosat stanford_cars \
  --model-name ViT-B-32 \
  --pretrained openai

python3 scripts/run_baseline.py \
  --datasets dtd food101 oxford_pets flowers102 fgvc_aircraft eurosat stanford_cars \
  --methods zero_shot prompt_ensemble linear_probe ridge svm nearest_centroid knn soft_knn tip_adapter ape lpplusplus gda_clip proker \
  --shots 1 2 4 8 16 \
  --seeds 1 2 3 \
  --model-name ViT-B-32 \
  --pretrained openai
```

To switch the frozen backbone, change only `--model-name` and `--pretrained`,
for example:

```bash
python3 scripts/run_baseline.py --model-name ViT-B-16 --pretrained openai
python3 scripts/run_baseline.py --model-name ViT-L-14 --pretrained openai
python3 scripts/run_baseline.py --model-name ViT-SO400M-14-SigLIP --pretrained webli
```

Available checkpoint names depend on the installed `open_clip_torch` version.

## Target Datasets

The main ClipGame comparison currently focuses on seven standard CLIP few-shot datasets:

- DTD
- Food101
- OxfordPets
- Flowers102
- StanfordCars
- FGVCAircraft
- EuroSAT

The broader low-shot scaling comparison may additionally follow the 11-dataset protocol used by prior CLIP adaptation papers.

## Data Notes

StanfordCars is prepared under `data/stanford_cars/`. When the Stanford official
server is unavailable, use the local conversion script after downloading the
`tanganke/stanford_cars` Hugging Face parquet shards:

```bash
python3 scripts/prepare_stanford_cars_from_hf.py
```

The script writes a baseline-compatible `split_zhou_StanfordCars.json` for the
local image filenames and keeps the downloaded official Zhou split as
`split_zhou_StanfordCars.official_paths.json` when available.

## Suggested Repository Layout

```text
tfclip/
  README.md
  configs/
    datasets/
    methods/
    experiments/
  tfclip/
    data/
    features/
    prompts/
    methods/
    evaluation/
    utils/
  scripts/
    extract_features.py
    run_baseline.py
    run_sweep.py
    collect_results.py
  external/
    baselines/
      README.md
      sources.json
      clip/
      tip_adapter/
      ape/
      lpplusplus/
      gda_clip/
      proker/
  outputs/
    features/
    results/
    logs/
```

## Reproduction Strategy

1. Extract and cache CLIP image/text features once for every dataset split.
2. Implement simple baselines directly on cached features.
3. Port official implementations for Tip-Adapter, APE, LP++, GDA-CLIP, and ProKeR behind a common method interface.
4. Run all methods from the same config files.
5. Save per-dataset, per-shot, per-seed metrics in machine-readable files.
6. Aggregate results into tables matching the paper.

## Method Naming

Use the following umbrella term in reports and paper text:

> frozen-feature few-shot CLIP adaptation baselines

This term includes zero-shot prompting, prompt ensembles, shallow classifiers, metric classifiers, cache-based adaptation, Gaussian discriminant analysis, and kernel-based adaptation while excluding gradient-based prompt/adaptor tuning methods.
