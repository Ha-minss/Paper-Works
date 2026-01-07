# paper_outputs 산출물 체크리스트 (v1.1 DoD)

이 체크리스트는 **후임자 구현 완료 여부를 판정**하기 위한 고정 규약입니다.

## DoD 실행 명령 (질문 없이 바로 동작해야 함)

아래 한 줄이 **추가 입력/질문 없이** 끝까지 실행되어야 합니다.

```bash
python -m stabilitycheck.runner --demo --all --split --largeD --testify
```

## 폴더 규약

- 기본 출력 루트: `paper_outputs/<exp_id>/`
- Tier별 산출물은 하위 폴더에 저장: `paper_outputs/<exp_id>/<tier>/`

`exp_id` 추천 형식: `YYYYMMDD_HHMMSS_<dataset>_seed<seed>`

---

## A. 공통 (Tier 실행 1회당 기본 산출물)

각 Tier 폴더 `paper_outputs/<exp_id>/<tier>/` 에서 아래 파일들이 **항상 생성**되어야 합니다.

1. `design_table.csv`
   - 최소 컬럼: `spec_key, family, n, psi_hat, U, p, g_vec, g_scalar, msg, Q_json` + feature columns

2. `tier_summary.csv`
   - 한 줄 요약: `tier, n_total, n_ok, n_fail, K_rule, S_wc_mainK, max_g_mainK, Neff_mainK, lambda_mainK`

3. `Kcurve.csv`
   - 컬럼: `K, lambda, Neff, S_wc, max_g, n_ok_designs, converged, rel_err`

4. `Kcurve.png`

5. `wcurve.csv` (w-curve 사용 시)
   - 컬럼: `w2, w3, K, S_wc, max_g`

6. `wcurve.png` (w-curve 사용 시)

7. `softmax_qc.csv`
   - 컬럼: `K, lambda, Neff, rel_err, iters, converged, bracket_lo, bracket_hi`

---

## B. Split/Cross-fit Stage1 (split=True)

Tier 폴더에 아래 파일들이 추가로 생성되어야 합니다.

8. `split_summary.csv`
   - 컬럼: `fold, n_train, n_test, K, lambda_train, Neff_train, S_wc_test, max_g_test`

9. `Kcurve_split.csv`
   - 컬럼: `K, S_wc_split_mean, S_wc_split_se, max_g_split_mean`

10. `Kcurve_split.png`

11. `weights_train_topshare.csv`
   - 컬럼 예시: `fold, top_frac, weight_share`

---

## C. Large-|D| (largeD=True)

Tier 폴더에 아래 파일들이 추가로 생성되어야 합니다.

12. `LargeD_sampling_plan.json`

13. `LargeD_tightening.csv`
   - 컬럼: `iter, stage, n_designs, S_wc, max_g, coverage_p50, coverage_p90`

14. `LargeD_tightening.png`

15. `coverage.json`

---

## D. Stage1 testification (testify=True)

exp 루트 `paper_outputs/<exp_id>/` 에 아래 파일들이 생성되어야 합니다.

16. `meta_run.json`
   - 실행 메타: timestamp, exp_id, seed, dataset, tiers, runtime

17. `config_snapshot.json`
   - EngineConfig/SchemaConfig/Runner 설정 덤프

18. `Stage1_test_report.csv`
   - 컬럼: `tier, K, S_obs, critical_value, p_value_mc, alpha, B_used, decision, null_mode`

19. `Stage1_null_hist.png`

20. `size_power_table.csv`
   - 최소 3행: stable null / local violation / regime shift

---

## 최종 완료 정의 (DoD 한 줄)

`pytest -q` 통과 + `python -m stabilitycheck.runner --demo --all --split --largeD --testify` 한 번 실행으로,
A(1~7) + B(8~11) + C(12~15) + D(16~20)가 **누락 없이 동일한 파일명**으로 생성되면 merge.
