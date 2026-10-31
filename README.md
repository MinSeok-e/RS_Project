# Recommender System Project
2026713004 / Minseok Jeong

EASE, SLIM ElasticNet, RP3β, and SASRec on the competition dataset.

- Per-user chronological ~80/20 split; all observed interactions = 1.
- Full-catalog NDCG@20: exclude seen items; cold validation targets remain in IDCG.
- SASRec uses fit history only. RP3β uses row top-k and row normalization.
- Selected settings: `configs/best.json` (default).

Run on the Slurm login node:
```bash
bash scripts/submit_compare.sh smoke
bash scripts/submit_compare.sh cpu
bash scripts/submit_compare.sh gpu
bash scripts/submit_refit.sh  # all train.csv rows; configs/submission.json
```
Jobs use `/home/minseok/.venvs/rs-hw2/bin/python` and the competition `../data` directory.
Kaggle API: `sbatch slurm/kaggle.sbatch submit --manifest results/full_JOB_ID/manifest.json`.
For the original searches, set `RS_CONFIG=configs/compare.json` or `configs/slim_regularization.json`.
Snapshots, logs, and results stay local; commit model code/settings at Kaggle submission time.

References: [EASE](https://arxiv.org/abs/1905.03375), [SLIM/RP3β implementation reference](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation), [SASRec](https://arxiv.org/abs/1808.09781).
