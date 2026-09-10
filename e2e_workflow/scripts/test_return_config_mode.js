#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Execute the shipped setup/resume, carry-state and accepted-config emitters.
'use strict';
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
function between(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert(a >= 0 && b > a, `workflow source block: ${start}`);
  return source.slice(a, b);
}
const init = between('const INIT_FLAGS =', '// Schema-v2 handoffs');
const setupInputs = between('      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT,', "      MEASUREMENT_PURPOSE: 'parity'");
const setup = between('  // An explicitly complete seed', '  curOverlay = INIT_BASE_OVERLAY;');
const resume = between("  curFlags = ST.flags || '';", '  curOverlay = ST.overlay || INIT_BASE_OVERLAY;');
const state = between('const carryState = {', '// What this run got from the KB');
const accepted = between('\n  accepted_config: { flags: curFlags', '  accepted_kernels: acceptedKernels,');

function run(args = {}, carried = null, setupFlags = '--disable-cuda-graph') {
  const context = {
    A: args, ST: carried || {}, setup: { server_flags: { extra: setupFlags }, server_env: 'RECIPE=1' },
    curArgsMode: 'append', curFlags: '', curEnv: '', curUnsetEnvs: [], curRemoveArgs: [], curOverlay: '', curTput: 100,
    BACKEND: 'sglang', EVAL_DIR: '/eval', MODEL_NAME: 'test', BASELINE_TPUT: 100,
    NOISE_BAND: 0.5, profile: {}, strategy: {}, headQueue: [], kernelQueue: [],
    acceptedHeads: [], flaggedHeads: [], acceptedKernels: [], tuning: null,
    pendingIntegrations: [], history: {},
    LAUNCH_SCRIPT: '/recipe', MODEL_PATH: '/model', EXP_ROOT: '/exp', EVAL_DIR_OVERRIDE: '',
    MODEL_NAME_HINT: 'test', TASK: '', GPU_IDS: '0', WORKLOAD: {}, INIT_BASE_OVERLAY: '',
  };
  const script = `${init}\n${carried ? resume : setup}\n${state}\n` +
    `JSON.stringify({ state: carryState, ${accepted}
      setup_inputs: ${carried ? 'null' : `({ ${setupInputs} })`} });`;
  return JSON.parse(vm.runInNewContext(script, context, { timeout: 1000 }));
}

function test() {
  const flags = '--context-length 9728 --cuda-graph-max-bs 64';
  const complete = run({ initial_extra_server_args: flags, initial_args_mode: 'replace' });
  assert.deepEqual(complete.accepted_config, { flags, env: 'RECIPE=1', args_mode: 'replace', remove_args: [] });
  assert(!complete.accepted_config.flags.includes('--disable-cuda-graph'));
  assert.equal(complete.state.args_mode, 'replace');
  assert.equal(complete.setup_inputs.INIT_ARGS_MODE, 'replace');
  assert.equal(complete.setup_inputs.INIT_FLAGS, flags);

  const resumed = run({ initial_extra_server_args: '--different-seed', initial_args_mode: 'append' }, complete.state);
  assert.deepEqual(resumed.accepted_config, complete.accepted_config,
    'a later invocation uses the carried flags and their completeness together');

  const empty = run({ initial_extra_server_args: '', initial_args_mode: 'replace' });
  assert.equal(empty.accepted_config.flags, '', 'a complete empty base cannot restore setup recipe flags');
  assert.equal(empty.accepted_config.args_mode, 'replace');
  assert.equal(empty.setup_inputs.INIT_ARGS_MODE, 'replace');
  assert.equal(empty.setup_inputs.INIT_FLAGS, '', 'the baseline-measuring role receives the explicit empty seed');

  const legacy = run({ initial_extra_server_args: '--candidate-only' });
  assert.deepEqual(legacy.accepted_config, { flags: '--candidate-only', env: 'RECIPE=1' });
  assert(!Object.hasOwn(legacy.state, 'args_mode'));
  assert(!Object.hasOwn(legacy.setup_inputs, 'INIT_ARGS_MODE'));
  const legacyResume = run({ initial_args_mode: 'replace' }, { flags: '--saved-delta', env: '' });
  assert.deepEqual(legacyResume.accepted_config, { flags: '--saved-delta', env: '' },
    'a newly complete handoff must not relabel an older carried delta');
  assert(!Object.hasOwn(legacyResume.state, 'args_mode'));

  const standalone = run();
  assert.deepEqual(standalone.accepted_config, { flags: '--disable-cuda-graph', env: 'RECIPE=1' });
  const invalidMode = run({ initial_args_mode: 'unknown', initial_extra_server_args: '--candidate-only' });
  assert(!Object.hasOwn(invalidMode.accepted_config, 'args_mode'));
  const literal = run({ initial_args_mode: 'replace', initial_extra_env: "'JSON={\"x\": \"space value\"}' EMPTY=" });
  assert.equal(literal.accepted_config.env, "'JSON={\"x\": \"space value\"}' EMPTY=",
    'argument completeness does not reinterpret environment assignments');
  const unset = run({ initial_extra_env: '', initial_env_complete: true,
    initial_unset_envs: ['RECIPE'], initial_args_mode: 'replace' });
  assert.equal(unset.accepted_config.env, '', 'complete empty env cannot restore setup assignments');
  assert.deepEqual(unset.accepted_config.unset_envs, ['RECIPE']);
  assert.deepEqual(unset.state.unset_envs, ['RECIPE']);
  assert.equal(unset.setup_inputs.INIT_ENV_COMPLETE, true);
  assert.deepEqual(unset.setup_inputs.INIT_UNSET_ENVS, ['RECIPE']);
  assert.deepEqual(run({}, unset.state).accepted_config, unset.accepted_config);
  assert(!Object.hasOwn(run({ initial_unset_envs: ['NEW'] }, { flags: '', env: '' }).accepted_config, 'unset_envs'),
    'older carried state does not acquire a new handoff removal');
  const removed = run({ initial_extra_server_args: '--keep 1', initial_args_mode: 'replace',
    initial_remove_args: ['--disable-radix-cache'] });
  assert.deepEqual(removed.accepted_config.remove_args, ['--disable-radix-cache']);
  assert.deepEqual(run({}, removed.state).accepted_config, removed.accepted_config);
  assert(!Object.hasOwn(run({ initial_remove_args: ['--new'] }, { flags: '', env: '' }).accepted_config, 'remove_args'));
  console.log('12 shipped workflow configuration transport cases passed');
}

module.exports = { run };
if (require.main === module) test();
