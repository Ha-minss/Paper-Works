# 후임자에게 보내는 "완벽 지시문" (stabilitycheck v1.1)

아래 내용은 그대로 복사해서 전달하면 됩니다. 핵심은 **구현만 하고 끝내지 말고**, `runner` 기본 플로우에 통합되어 **고정된 파일명/컬럼 스키마로 산출물이 자동 생성**되도록 강제하는 것입니다.

---

## 0) 전제: 변경 금지 (Non-negotiables)

1) `stabilitycheck/engine.py`는 **계산만** 한다. (그림 저장/파일 I/O 금지)

2) `stabilitycheck/runner.py`가 **논문 산출물 전부를 저장**한다. (폴더/파일명 규약 유지)

3) `EstimationResult` 인터페이스/필드명은 **절대 변경 금지**:

- `ok, psi_hat, U, p, Q, n, msg`

4) 내부 표준

- `U`는 **covariance matrix (k×k)** 가 표준이다.
  - 만약 어댑터가 `se` 벡터를 주면 `as_cov(se)`로 `diag(se^2)`로 승격한다.
- 기본 네이밍 유지: `S_wc`, `Neff`, `K_rule`, `mu`(= mu_adv), `g_vec`, `g_scalar`

---

## 1) [필수] Stage1 split/cross-fit (post-search validity 봉인)

### 절대 규칙

- `train` fold에서만 `mu_adv`(=weights) 학습
- `test` fold에서는 **mu 고정 적용**
  - **λ 재추정 금지**
  - **mu 재계산 금지**

### test fold score 정의

- `S_wc_test = sum_d mu_d_train * g_d_test`

### 완료 기준(DoD)

`runner` 실행 시 아래 파일이 반드시 생성:

- `paper_outputs/<exp_id>/<tier>/Kcurve_split.csv`
- `paper_outputs/<exp_id>/<tier>/Kcurve_split.png`
- `paper_outputs/<exp_id>/<tier>/split_summary.csv`
- `paper_outputs/<exp_id>/<tier>/weights_train_topshare.csv`

추가 조건:

- `run_all(..., split=True)`가 `--demo` 데이터로 **1분 내 스모크 통과**

---

## 2) [필수] K→λ root-finding + QC (softmax.py)

### 절대 규칙

- 입력은 `lambda`가 아니라 `K`다.
- `Neff(lambda) = K` 되도록 자동 브라켓 확장 + bisection(or brent)
- QC 실패 시:
  - `converged=False` + 경고 로그
  - fallback λ 사용(경계값)하되 **프로세스는 죽지 않게**

### 완료 기준(DoD)

- `tests/test_softmax.py`에 3개 테스트:
  1) λ 증가 시 Neff 증가(단조)
  2) 극단 케이스(평평 g / 스파이크 g)에서도 동작
  3) `rel_err < 1e-3` 수렴

- `runner` summary에 `lambda, Neff, converged, rel_err`가 항상 기록됨
- `paper_outputs/<exp_id>/<tier>/softmax_qc.csv`가 항상 생성됨

---

## 3) [필수] Large-|D|: coverage/refine/sweep + tightening report

### 절대 규칙

- largeD는 "옵션 파일만 존재"하면 안 됨.
  - `runner.run_tier(..., largeD=True)`에서 자동 실행 & 자동 저장
- tightening report는 **필수 산출물**:
  - iter별 `S_wc, max_g, coverage` 기록

### 완료 기준(DoD)

반드시 생성:

- `paper_outputs/<exp_id>/<tier>/LargeD_sampling_plan.json`
- `paper_outputs/<exp_id>/<tier>/LargeD_tightening.csv`
- `paper_outputs/<exp_id>/<tier>/LargeD_tightening.png`
- `paper_outputs/<exp_id>/<tier>/coverage.json`

추가 조건:

- `refine` + `sweep` 단계가 최소 1회 실행되는 로그가 남아야 함

---

## 4) [필수] Stage1 testification 강화 (null/testification/dgp)

### 절대 규칙

- Stage1을 "점수"가 아니라 "검정"으로 출력:
  - critical value `c_alpha`
  - Monte Carlo p-value `p_mc`
  - decision `PASS/FAIL`

- 최소 3종 DGP 템플릿:
  1) stable null (size)
  2) local violation (power)
  3) regime shift (power)

### 완료 기준(DoD)

반드시 생성:

- `paper_outputs/<exp_id>/Stage1_test_report.csv`
- `paper_outputs/<exp_id>/Stage1_null_hist.png`
- `paper_outputs/<exp_id>/size_power_table.csv`

추가 조건:

- `B=200` 스모크 통과
- `B=1000`도 안정 실행 (병렬 optional)

---

## 5) 최종 완료 정의(한 줄)

`pytest -q` 통과 + `python -m stabilitycheck.runner --demo --all --split --largeD --testify` 1회 실행으로
`paper_outputs/<exp_id>/` 아래에 DoD 체크리스트 파일들이 **고정 파일명으로 누락 없이 생성되면 merge**.

