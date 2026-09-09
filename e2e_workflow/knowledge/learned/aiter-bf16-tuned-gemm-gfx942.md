---
key: dense bf16 GEMM · gfx942 · sglang
type: lever
confidence: ★★★
effect: +2.23% e2e VERIFIED (1548.9→1583.5 tok/s, non-overlapping 5-repeat A/B); ~+6% cumulative w/ attn-triton
confirms: 2
last_seen: 2026-06-08
---
# Dense bf16 GEMM → tune aiter's per-shape DB (the #1 verified e2e win on this stack)
- lever: the live dense-GEMM path on sglang/gfx942 is aiter `tuned_gemm.py` (executing hipBLASLt
  `Cijk_*`), seam `aiter.tuned_gemm:gemm_a16w16`. Tune its per-shape DB — this is THE GEMM lever, and
  the strongest *transferred-to-e2e* win recorded.
- apply: capture real shapes `AITER_TUNE_GEMM=1` → `gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>`
  → deploy `AITER_CONFIG_GEMM_BF16=<tuned.csv>` (pure env, no package edit). FlyDSL races inside this DB
  (`libtype=flydsl`) and is auto-selected where it wins.
- caution (gfx950 · MI355X · DSV4, VERIFIED 2026-09-09): aiter's OWN default table can kill the server
  at first inference. With `AITER_CONFIG_GEMM_BF16` unset, `get_config_file` merges
  `configs/bf16_tuned_gemm.csv` with `configs/model_configs/*bf16_tuned_gemm*.csv` into
  `/tmp/aiter_configs/bf16_tuned_gemm.csv` (112 → 3007 rows, 985 of them `libtype=flydsl`). DSV4's
  `compressor.compute_kv_score` issues `m=6,n=1024,k=7168`; the padded-M lookup lands on the merged
  table's flydsl row `gemm5_t16x64x128_split_k8...gfx950`, and `_validate_hgemm_tiling` rejects it →
  `ValueError: Invalid tiling configuration for m=6 n=1024 k=7168` → server dies at boot-time inference.
  Probed both ways on the live image: env unset → flydsl (crashes); env set to a tuned CSV → the shape
  misses and falls to `libtype=torch` (safe), because a SINGLE path short-circuits
  `update_config_files` and the flydsl-carrying `model_configs/` are never merged in.
  So deploy the tuned CSV **as a single path** (do NOT union the vendored/model_configs tables back in —
  that reintroduces the crashing row), and deploy it BEFORE the first server launch of the phase: an
  early launch that predates the tuned CSV runs on the default table and eats a full boot.
- verify: `AITER_LOG_TUNED_CONFIG=1` → count `is tuned on cu_num` hits (>0 = engaged; the winning run
  had 246 hits). The capture's correct `bias=False` + full shape coverage is what makes it both ENGAGE
  and WIN — a bias-mismatched/partial tune reads ~0/−0.6% (superseded).
- caution: NOT TunableOp / `HIPBLASLT_TUNING_FILE` — aiter bypasses the PyTorch/hipBLASLt C dispatch
  for its tuned shapes, so those hooks don't touch the live path.
- caution (STACK-SPECIFIC, verify the seam): this lever assumes the live GEMM goes through
  `aiter.tuned_gemm:gemm_a16w16`. On **vLLM** (≥0.19, gfx942) the dense bf16 path is
  `vllm...layers.utils:rocm_unquantized_gemm_impl`, which for these Qwen3-14B (N,K) families routes to
  `torch.nn.functional.linear` → hipBLASLt and **does not call aiter.tuned_gemm** (`use_aiter_triton_gemm()`
  returns False for all 4 families on gfx942; on_gfx950 paths off). So `AITER_CONFIG_GEMM_BF16` would get
  **0 engagement on vLLM** — confirm the seam before tuning. There, Tier-B yields nothing (TunableOp ties
  box-default hipBLASLt isolated); the lever is the **Tier-C author** route rebound at
  `rocm_unquantized_gemm_impl` (existing editable `aiter.ops.triton.gemm_a16w16` is a good baseline, but
  gated OFF for these shapes). [gfx942 · vLLM dense bf16, 2026-06-22 Qwen3-14B]
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-08 (verified A/B, full recipe in `SKILL_DIR/knowledge/gemm_tuning/aiter_gemm_tuning.md`);
  vLLM-seam caution: exp/e2e_Qwen-Qwen3-14B_20260622 bake-off (seam-inspected, 0-engagement predicted)
