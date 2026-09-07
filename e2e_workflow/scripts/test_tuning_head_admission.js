#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Execute the shipped tuning and head scheduling blocks with deterministic workers.
// These are CPU orchestration tests, not GPU measurements or cancellation tests.
'use strict';
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
const between = (start, end) => {
  const a = src.indexOf(start), b = src.indexOf(end, a);
  assert(a >= 0 && b > a, `source block exists: ${start}`);
  return src.slice(a, b);
};
const budgetSrc = between('const TIME_BUDGET_MS =', '// ---- FAST MODE');
const tuneSrc = between('// PHASE: TuningSkillset', '// PHASE: HeadKernel');
const headSrc = between("if (want('head') && headQueue.length && HEAD_BUDGET > 0)", '// PHASE: Milestone loop');
const returnSrc = between('function tuningReturn()', 'const wfReturn =');

async function runCase(options = {}) {
  const A = { ...(options.args || {}) };
  if (options.budgetS !== null) A.time_budget_s = options.budgetS === undefined ? 15942 : options.budgetS;
  const budget = new Function('A', 'setTimeout', budgetSrc + `
    return { TIME_BUDGET_MS, TIME_HEAD_DEADLINE_MS, FINAL_RESERVE_MS };
  `)(A, () => ({ unref() {} }));
  const heads = options.noHeads ? [] : [{ short_name: 'attention', op_kind: 'attention', pct_gpu_time: 9 }];
  const phases = new Set(options.phases || ['tune', 'head']);
  const calls = [], acquired = [], released = [], logs = [];
  let active = 0;
  const carried = options.carriedTuning || null;
  const ctx = {
    A, ST: carried ? { tuning: carried } : {}, tuning: carried, ...budget,
    ELAPSED_MS: options.expired ? 10071000 : 3600000,
    TIME_DEADLINE_HIT: !!options.expired, FAST_DEADLINE_HIT: false,
    want: (phase) => phases.has(phase), phase: (name) => calls.push('phase:' + name),
    log: (line) => logs.push(line),
    TUNING_SKILLSET_ENABLED: String(A.tuning_skillset == null ? true : A.tuning_skillset) === 'true',
    TUNING_SKILLSET_DIR: '/skills', TUNING_KB_ENABLED: false,
    HEAD_BUDGET: options.headBudget == null ? 1 : options.headBudget,
    HEAD_AUTHOR_MAX: 1, HEAD_PROTECT_PCT: 30, HEAD_THRESHOLD_PCT: 5,
    FAST_MODE: false, DEEP: false, headQueue: heads, kernelQueue: [], headDispatched: 0,
    history: { ledger: [] }, acceptedHeads: [], acceptedKernels: [], flaggedHeads: [],
    EVAL_DIR: '/eval', MODEL_PATH: '/model', WORKFLOW_DIR: '/workflow',
    GPU_LIST: ['0'], WORKLOAD: { isl: 8192, osl: 1024, conc: 64 }, CONC: 64,
    BASELINE_TPUT: 1000, curTput: carried ? 1100 : 1000,
    curEnv: carried ? carried.apply_env : '', curFlags: '',
    curOverlay: carried ? carried.apply_overlay : '/overlay/base',
    profile: {}, strategy: {}, SEARCH_REPLICAS: 1, NOISE_BAND: 0.5, E2E_REPEATS: 1,
    ACCURACY_GATE: 'none', ANALYSIS_SKILL_INPUTS: {}, TUNING_SCHEMA: {}, PROFILE_SCHEMA: {},
    STRATEGY_SCHEMA: {}, EXTRACT_OP_SCHEMA: {}, OPBENCH_SCHEMA: {}, KB_DIMS: null,
    ENABLE_FP8: true, KERNEL_WF_DIR: '/kernel', KERNEL_WF_SCRIPT: '/kernel/workflow.js',
    KERNEL_KNOWLEDGE_DIR: '/knowledge', KERNEL_BUDGET: 1, BUDGET: 1,
    CONFIG_TUNE_ENABLED: false, USE_EXPERT_SKILLS: false,
    EXPERT_SKILLS_DIR: '/experts', KB_ARGS: {}, GRAPH_REQ: '', TASK: '',
    validatedOk: false, validation: null, finalize: null, finalTput: 0,
    roleAgent: (role, phase, intro, inputs) => ({ role, phase, inputs }),
    gemmSynthFor: () => false,
    applyOpIdentityGuard: (queue) => queue,
    ensureFlydslGate: async () => {},
    bankAccepted: (entries, record) => entries.push(record),
    e2eFrom: (result) => ({ e2e_delta_pct: result.e2e_delta_pct }),
    krOf: (inputs) => inputs.KERNEL_RESULT,
    integAccepted: (result) => result.gate === 'accepted',
  };
  const advance = (ms) => {
    ctx.ELAPSED_MS += ms;
    ctx.TIME_DEADLINE_HIT = budget.TIME_HEAD_DEADLINE_MS != null &&
      ctx.ELAPSED_MS >= budget.TIME_HEAD_DEADLINE_MS;
  };
  const worker = async (name, duration, result) => {
    calls.push(name); acquired.push(name); active++;
    try { advance(duration); return result; }
    finally { active--; released.push(name); }
  };
  ctx.safeAgent = async (prompt) => {
    if (prompt.role === 'tuning_specialist') {
      // The historical receipt bounds combined pre-head work at 10,070.79 seconds.
      // Phase-entry time here is a simulation input, not a recovered timestamp.
      const duration = options.slowTuning ? 10070793 - ctx.ELAPSED_MS : 1000;
      return worker('tuning', duration, options.acceptTuning ? {
        gate: 'accepted', engagement_verified: true, ab_complete: true,
        correctness_gate: 'pass', pre_tune_throughput_tok_s: 1000,
        post_tune_throughput_tok_s: 1100, tuning_delta_pct: 10,
        apply_env: 'TUNED_TABLE=/eval/tuned.csv', apply_overlay: '/overlay/tuned',
        deploy_bundle: '/eval/tuning/deploy', deploy_verified: true,
        ops_tuned: [{ op: 'gemm', backend: 'aiter', isolated_speedup: 1.1, engaged: true }],
      } : { gate: 'no_win', reason: 'no measured tuning win' });
    }
    if (prompt.role === 'op_benchmarker') return worker('bakeoff', 1000, {
      gate: 'author_recommended', author_plan: [{ language: 'triton' }],
    });
    if (prompt.role === 'system_architect') return worker('strategize', 1000, { head_candidates: heads });
    return worker('profile', 1000, { profile_topN_json: '/profile.json' });
  };
  ctx.extractWithBaseline = async (_role, _phase, _intro, inputs) => {
    ctx.extractionInputs = inputs;
    return worker('capture', 1000, { smoke: 'pass', task_dir: '/task', op_kind: 'attention' });
  };
  let authorAttempts = 0;
  ctx.fastBoundedWorkflow = async (_ref, args) => {
    assert.equal(args.mode, 'author');
    authorAttempts++;
    if (options.retryAuthor && authorAttempts === 1)
      return worker('author-transient', 1000, { authored: false, validation_status: 'error' });
    return worker('author', 1000, { authored: true, final_geomean: 1.2,
      final_patch: '/head.patch', eval_dir: '/head' });
  };
  ctx.runIntegrateBothLegs = async (_intro, inputs) => {
    ctx.integrationInputs = inputs;
    return worker('integrate', 1000, { gate: 'accepted', ab_complete: true,
      ref_med: ctx.curTput, e2e_throughput_tok_s: ctx.curTput + 20,
      e2e_delta_pct: 2, accepted_overlay: '/overlay/combined' });
  };
  // Negative control removes ONLY the admission decision from the executed source.
  // The rest of the shipped tuning and head dispatch paths are identical.
  const tuningCode = options.withoutAdmission
    ? tuneSrc.replace(' && !tuningAdmissionSkip)', ')') : tuneSrc;
  const script = new vm.Script(`(async () => {
    ${tuningCode}
    ${headSrc}
    finalTput = curTput;
    ${returnSrc}
    return { tuning, result: tuningReturn(), finalizeInputs: TUNING_FINALIZE_INPUTS,
      reportInputs: TUNING_REPORT_INPUTS };
  })()`);
  const result = await script.runInNewContext(ctx, { timeout: 1000 });
  assert.equal(active, 0, 'all simulated workers returned; the skip path starts none');
  assert.deepEqual(acquired, released, 'no simulated worker is abandoned by a timer race');
  return { ...result, ctx, calls, authorAttempts, logs };
}

(async () => {
  const before = await runCase({ slowTuning: true, withoutAdmission: true });
  assert(before.calls.includes('tuning'));
  assert.equal(before.authorAttempts, 0, 'the old scheduling path loses all author dispatch');
  const after = await runCase({ slowTuning: true, retryAuthor: true });
  assert(!after.calls.includes('tuning'), 'implicit tuning starts no worker or cleanup obligation');
  assert.equal(after.authorAttempts, 2, 'capture, bakeoff and a transient retry reach authoring');
  assert.equal(after.ctx.acceptedHeads.length, 1);
  assert.equal(after.result.ran, false);
  assert.equal(after.result.gate, 'skipped');
  assert.equal(after.result.reason, 'implicit_tuning_skipped_for_head_generation');
  assert.equal(after.ctx.history.ledger[0].reason, after.result.reason);
  assert.equal(after.reportInputs.TUNING_ADMISSION_SKIP.reason, after.result.reason);
  assert.equal(after.result.admission_skip.queued_heads, 1);
  assert.equal(after.ctx.FINAL_RESERVE_MS, before.ctx.FINAL_RESERVE_MS);
  assert.equal(after.ctx.TIME_HEAD_DEADLINE_MS, before.ctx.TIME_HEAD_DEADLINE_MS);

  const combined = await runCase({ args: { tuning_skillset: true }, acceptTuning: true });
  assert(combined.calls.indexOf('tuning') < combined.calls.indexOf('capture'));
  assert.equal(combined.ctx.acceptedKernels.length, 1, 'tuned table remains banked');
  assert.equal(combined.ctx.acceptedHeads.length, 1, 'new authored overlay remains banked');
  assert.equal(combined.ctx.extractionInputs.CURRENT_ENV, 'TUNED_TABLE=/eval/tuned.csv');
  assert.equal(combined.ctx.integrationInputs.CURRENT_OVERLAY, '/overlay/tuned');
  assert.equal(combined.finalizeInputs.TUNING_DEPLOY_BUNDLE, '/eval/tuning/deploy');
  assert.equal(combined.result.gate, 'accepted');

  const carried = { ...combined.tuning };
  const resumed = await runCase({ carriedTuning: carried });
  assert(!resumed.calls.includes('tuning'));
  assert.deepEqual(resumed.tuning, carried, 'skipping a new attempt preserves the prior accept');
  assert.equal(resumed.ctx.extractionInputs.CURRENT_ENV, carried.apply_env);
  assert.equal(resumed.finalizeInputs.TUNING_DEPLOY_BUNDLE, carried.deploy_bundle);
  assert.equal(resumed.result.gate, 'accepted');
  assert.deepEqual(resumed.reportInputs.TUNING_RESULT, carried);
  assert.equal(resumed.result.admission_skip.reason, after.result.reason);

  for (const options of [{ noHeads: true }, { headBudget: 0 },
    { phases: ['tune'] }, { budgetS: null }, { budgetS: 0 },
    { budgetS: -1 }, { budgetS: 'Infinity' }, { budgetS: 'invalid' }]) {
    const result = await runCase(options);
    assert(result.calls.includes('tuning'), JSON.stringify(options));
    assert(!result.result.admission_skip);
  }
  const disabled = await runCase({ args: { tuning_skillset: false } });
  assert(!disabled.calls.includes('tuning'));
  assert.equal(disabled.result.enabled, false);
  const expired = await runCase({ expired: true });
  assert(!expired.calls.includes('tuning'));
  assert.equal(expired.authorAttempts, 0, 'admission cannot recover an already-expired deadline');
  const explicitSlow = await runCase({ args: { tuning_skillset: 'true' }, slowTuning: true });
  assert(explicitSlow.calls.includes('tuning'));
  assert.equal(explicitSlow.authorAttempts, 0, 'explicit tune-first retains its documented tradeoff');
  console.log('PASS: source-derived admission, starvation control, author dispatch/retry, combined win, carried state, phase/budget boundaries, and no spawned tuning worker on skips.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
