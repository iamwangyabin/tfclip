# External Baseline Source Snapshots

This directory vendors upstream source snapshots for the primary baseline
reproduction work. The snapshots were fetched on 2026-06-10 and copied without
their nested `.git` directories so they can be tracked by this repository.

These trees are references for reproduction and porting. The unified runner
should call project-local code under `tfclip/` and only depend on these
directories while we are auditing or migrating the official implementations.

## Snapshot Index

| Directory | Upstream | Commit | Used for |
|---|---|---:|---|
| `clip/` | https://github.com/openai/CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | Zero-shot CLIP, prompt ensemble text/image encoding, linear-probe reference |
| `tip_adapter/` | https://github.com/gaopengcuhk/Tip-Adapter | `d0e2d6f8c5feb8b6ce937b757810761f7155d4d5` | Tip-Adapter and Tip-Adapter-F |
| `ape/` | https://github.com/yangyangyang127/APE | `70460de74d1e967f8b4d270b3419373694758536` | APE |
| `lpplusplus/` | https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline | `f81054f995b060f8348d20e7306e15889b70c786` | LP++ |
| `gda_clip/` | https://github.com/mrflogs/ICLR24 | `560fe211700964c915442ff98dbe47dd09656a35` | GDA-CLIP |
| `proker/` | https://github.com/ybendou/ProKeR | `f2ecb3d2a0d7e0993f977e550286b34f06015edb` | ProKeR |

## Baseline Mapping

| Method | Source in this repo | Integration note |
|---|---|---|
| Zero-shot CLIP | `external/baselines/clip/` | Implement in the unified runner using frozen CLIP text/image features. |
| Prompt ensemble | `external/baselines/clip/` | Average normalized text embeddings over prompt templates. |
| Linear probe | `external/baselines/clip/` plus scikit-learn | Use frozen image features and `sklearn.linear_model.LogisticRegression`. |
| Ridge classifier | scikit-learn dependency | Implement directly with `sklearn.linear_model.RidgeClassifier`. |
| Linear SVM | scikit-learn dependency | Implement directly with a linear SVM on frozen image features. |
| Nearest centroid | scikit-learn dependency | Implement directly with `sklearn.neighbors.NearestCentroid`. |
| kNN / soft kNN | scikit-learn dependency plus local voting code | Use `KNeighborsClassifier` for hard kNN and local cosine-weighted voting for soft kNN. |
| Tip-Adapter | `external/baselines/tip_adapter/` | Port cache construction and affinity logits to a common feature interface. |
| APE | `external/baselines/ape/` | Port the official adaptation logic while reusing our cached CLIP features. |
| LP++ | `external/baselines/lpplusplus/` | Port the text-informed linear-probe procedure. |
| GDA-CLIP | `external/baselines/gda_clip/` | Port Gaussian discriminant scoring on shared feature splits. |
| ProKeR | `external/baselines/proker/` | Port kernel/prototype logic to shared support/query features. |

## Maintenance Notes

- Keep upstream license and README files inside each vendored directory.
- Do not assume upstream dataset splits, prompt templates, or cached features
  match this project; normalize those through the unified runner.
- Record any future upstream refresh in `sources.json` and this README.
