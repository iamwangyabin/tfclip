# Official Baseline Calibration Notes

This document tracks the status of matching the vendored official
implementations while preserving this repository's unified `open_clip` feature
interface.

## Calibrated In Code

- `configs/methods/official_feature_space.yaml` is the single source of truth
  for method defaults. `tfclip.methods.config` loads it directly, so YAML and
  code cannot silently drift.
- Tip-Adapter: all 11 official dataset grids for `search_scale`,
  `search_step`, `init_alpha`, `init_beta`, `augment_epoch`, and `train_epoch`
  are copied from `external/baselines/tip_adapter/configs/*.yaml`.
- APE: all 11 official alpha/beta/gamma grids, feature-selection counts,
  `eps`, `w_training_free`, `w_training`, `augment_epoch`, and `train_epoch`
  are copied from `external/baselines/ape/configs/*.yaml`. APE also uses the
  official template-plus-CuPL text prompt cache.
- LP++: `LinearProbe_P2` follows `external/baselines/lpplusplus`: 300 epochs,
  one official train-transform support view, class-sum centroid initialization,
  feature/text-derived alpha initialization, and manual alpha updates every 10
  epochs.
- GDA-CLIP: `augment_epoch=10` and alpha grid
  `[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` match
  `external/baselines/gda_clip/main_few_shots.py` and few-shot configs.
- ProKeR: all 11 official beta/lambda grid ranges are copied from
  `external/baselines/proker/configs/RN50/configs_proker/*.yaml`, and
  `augment_epoch=10` matches the official CLI default.
- Support cache: official-cache methods store raw support encoder outputs keyed
  by dataset, model, seed, shots, augment epoch, transform policy, and raw
  feature space. Tip-Adapter and APE average raw augmented views and then
  normalize; LP++, GDA-CLIP, and ProKeR normalize each support view before
  fitting, matching their official estimator structure.
- `scripts/audit_method_calibration.py` checks the unified config against the
  vendored official configs and should pass before running paper tables.

The runner records selected hyperparameters in each JSONL result under
`best_params`.

## Deliberate Unified-Runner Differences

- Val/test image features still come from the unified `open_clip` backbone
  wrapper. Official repositories often re-encode these splits inside each
  method-specific script.
- Official support augmentation uses the paper-repo OpenAI CLIP transform:
  `RandomResizedCrop(224, scale=(0.5, 1.0), bicubic)`,
  `RandomHorizontalFlip(0.5)`, and OpenAI CLIP normalization. This is
  paper-repo consistent; for non-OpenAI or non-224 backbones it is intentionally
  not the model-specific `open_clip` train transform.
- The implemented APE path is the training-free APE variant. APE-T's trainable
  adapter is not enabled in the primary baseline list.
- StanfordCars currently uses a Hugging Face mirror converted to local paths
  because the Stanford official archive server returned HTTP 500 during setup.
  The local split has the same train/val/test counts but not the original
  Stanford file names.

## Next Calibration Step

Run per-paper numeric checks against the vendored official code on a small
dataset/model pair, then lock any remaining method-specific edge cases behind
explicit config switches rather than changing the shared feature protocol.
