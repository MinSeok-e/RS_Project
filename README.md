# Recommender System Project
2026713004 / Minseok Jeong

캐글에 제출한 모델 코드와 최종 설정만 `submission/`에서 관리합니다.

| 제출 | 모델 | 설정 |
|---|---|---|
| v3–v6 | EASE, SLIM, RP3β, SASRec | `submission/configs/submission.json` |
| v7 | SASRec CE, future-5 | `submission/configs/sasrec_submission.json` |
| v8 | Hybrid Ensemble + Reranking | `submission/configs/hybrid_ensemble_submission.json` |
| v9–v17 | MoE, FMLP, CatBoost 및 앙상블 | `submission/configs/v*.json` |

Slurm에서 실행:
```bash
bash submission/scripts/submit_refit.sh
RS_REFIT_CONFIG=configs/sasrec_submission.json bash submission/scripts/submit_refit.sh
bash submission/scripts/submit_hybrid_stage.sh bases
bash submission/scripts/submit_selected.sh v17.json
```
하이브리드는 bases → auxiliary → features → infer 순서로 실행하며, 이전 단계의 결과 경로를 전달합니다. 선택된 랭커 체크포인트와 학습 산출물은 서버에 보관합니다.
탐색 코드·후보 설정·데이터·로그·보고서는 GitHub에 올리지 않습니다.
