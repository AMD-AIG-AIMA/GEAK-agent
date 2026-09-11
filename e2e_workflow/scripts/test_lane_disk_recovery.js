#!/usr/bin/env node
// Regression guard for LANE DISK RECOVERY (no GPU, no model, no server needed).
//
// fixv5, 2026-09-10, DeepSeek-V4-Pro on node 318. The head kernel was picked correctly
// (`main_kernel`, the DSA sparse-MLA attention, 43.95% of prefill GPU time + 12.37% of decode) and
// team1 finished at 19:14 with a Director-arbitrated weighted 3.4199x / 3.4111x over two independent
// full unittest runs, correctness pass on every leg, `validation_status: "accepted"`. Its
// kernel_workflow RETURN was lost. The orchestrator read that loss as `transient`, started a SECOND
// team from baseline at 19:16, and the 21:13 deadline arrived with neither the owed e2e A/B nor the new
// team finished. The run shipped `throughput_speedup 1.0`, `accepted_kernels: []` -- a real 3.42x
// kernel thrown away because a return value did not survive, not because the work failed.
//
// The result was on disk the whole time: `exp_root/team_<op>_<ts>_*/<op>/final_patch.diff` plus
// `director_validation.json`. recoverLaneFromDisk() reads that truth. This test pins its contract:
//
//   1. it FINDS a finished team (patch + Director numbers) and carries the weighted speedup;
//   2. it carries `requires_e2e_confirmation` WITHOUT filtering on `timing_provenance_ok` -- per
//      kernel_workflow/roles/director.md, a merit-`accepted` lane with unproven provenance is eligible
//      to win the bake-off and MUST be carried to the e2e bench, because the e2e A/B *is* the
//      confirmation that unproven provenance owes;
//   3. it REFUSES a Director `rejected` lane (that verdict is on merit; recovery must not launder it);
//   4. it REFUSES a team with no patch / an empty patch / no Director number (no fabricated speedup);
//   5. it is SCOPED by the before-snapshot, so it can only pick up a team the current call created --
//      never another language's lane running concurrently under the same exp_root;
//   6. it never throws on a missing/unreadable exp_root -- it degrades to today's behaviour.
//
// The helpers are EXTRACTED from the real source, not reimplemented here: a copy would keep passing
// while the shipped code drifted.
//
// Run:  node e2e_workflow/scripts/test_lane_disk_recovery.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// ---------------------------------------------------------------- extract the real helpers
const src = fs.readFileSync(WORKFLOW, 'utf8');
const from = src.indexOf('const _teamDirs = (expRoot, opName)');
const isoAt = src.indexOf('const recoveredIso =');
if (from < 0 || isoAt < 0) {
  console.error('FAIL: could not locate the disk-recovery helpers in e2e_workflow.js — did they move or get deleted?');
  process.exit(1);
}
const block = src.slice(from, src.indexOf('\n', isoAt));
// `require` is a module-local binding, not a global, so hand it in explicitly.
const H = new Function('require', block +
  '\nreturn { _teamDirs, teamSnapshot, opNameOf, recoverLaneFromDisk, recoveredIso };')(require);

// ---------------------------------------------------------------- fixtures
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-lane-recovery-'));
const EXP = path.join(TMP, '_exp');
const OP = 'dsv4_sparse_mla_attn_task';
fs.mkdirSync(EXP, { recursive: true });

// A team as kernel_workflow leaves it: <exp_root>/team_<op>_<ts>_<pid>_<n>/<op>/{final_patch.diff,...}
const mkTeam = (stamp, { patch, validation }) => {
  const dir = path.join(EXP, `team_${OP}_${stamp}`, OP);
  fs.mkdirSync(dir, { recursive: true });
  if (patch != null) fs.writeFileSync(path.join(dir, 'final_patch.diff'), patch);
  if (validation) fs.writeFileSync(path.join(dir, 'director_validation.json'), JSON.stringify(validation));
  return dir;
};

// team1 verbatim from the fixv5 record that motivated this guard.
const TEAM1 = {
  kernel_name: OP,
  director_verified_speedup_geomean: 2.9877,
  director_verified_speedup_weighted: 3.4199,
  weighted_all_run1: 3.4199, weighted_all_run2: 3.4111,
  validation_status: 'accepted', correctness: 'pass',
  timing_basis: 'unknown', timing_provenance_ok: false,
  requires_e2e_confirmation: true, timing_receipt: {},
};

console.log('1. a finished team is recovered, with its Director number');
{
  const before = H.teamSnapshot(EXP, OP);
  const dir = mkTeam('20260910_121422_3187166_19228', { patch: 'diff --git a b\n+tilelang\n', validation: TEAM1 });
  const rec = H.recoverLaneFromDisk(EXP, OP, before);
  ok(!!rec, 'a team with a patch + director_validation.json is recovered');
  ok(rec && rec.eval_dir === dir, 'eval_dir points at the team workspace');
  ok(rec && rec.final_patch === path.join(dir, 'final_patch.diff'), 'final_patch points at the diff on disk');
  ok(H.recoveredIso(rec) === 3.4199, 'the WEIGHTED speedup is carried (3.4199), not the geomean');
  ok(rec && rec.final_geomean === 2.9877, 'the geomean is carried too, as the fallback basis');
  ok(rec && rec.validation_status === 'accepted', "the Director's own verdict is carried through");
  ok(rec && rec.recovered_from_disk === true, 'the entry is tagged recovered_from_disk (auditable provenance)');
  ok(rec && rec.authored === true, 'admitted on the authored path, like a live return');
}

console.log('2. unproven timing provenance is CARRIED to e2e, never filtered out (director.md contract)');
{
  const before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_140000_1_1', { patch: 'x', validation: TEAM1 });
  const rec = H.recoverLaneFromDisk(EXP, OP, before);
  ok(!!rec, 'timing_provenance_ok:false does NOT suppress recovery');
  ok(rec && rec.requires_e2e_confirmation === true, 'requires_e2e_confirmation is propagated to the caller');
  ok(rec && rec.timing_provenance_ok === false, 'the provenance flag itself is reported, not silently dropped');
}

console.log('3. a Director REJECTED lane stays rejected');
{
  const before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_150000_1_1', {
    patch: 'x', validation: { ...TEAM1, validation_status: 'rejected' },
  });
  ok(H.recoverLaneFromDisk(EXP, OP, before) === null, 'a merit rejection is never laundered into a candidate');
}

console.log('4. no patch / empty patch / no Director number => no candidate (never fabricate a speedup)');
{
  let before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_160000_1_1', { validation: TEAM1 });                      // director numbers, no diff
  ok(H.recoverLaneFromDisk(EXP, OP, before) === null, 'a team with no final_patch.diff is not a result');

  before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_161000_1_1', { patch: '', validation: TEAM1 });           // zero-byte diff
  ok(H.recoverLaneFromDisk(EXP, OP, before) === null, 'an empty diff is not a result');

  before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_162000_1_1', { patch: 'x' });                             // diff, no director_validation.json
  ok(H.recoverLaneFromDisk(EXP, OP, before) === null, 'a patch with no Director number claims no speedup');

  before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_163000_1_1', { patch: 'x', validation: { validation_status: 'accepted' } });
  ok(H.recoverLaneFromDisk(EXP, OP, before) === null, 'an accepted record with no measured speedup is still not a candidate');
}

console.log('5. the before-snapshot scopes recovery to teams THIS call created');
{
  const after = H.teamSnapshot(EXP, OP);   // everything above already exists
  ok(H.recoverLaneFromDisk(EXP, OP, after) === null,
    "a concurrent lane's finished team is never mis-attributed to this call");

  // Newest-wins among teams the call did create.
  const before = H.teamSnapshot(EXP, OP);
  mkTeam('20260910_170000_1_1', { patch: 'x', validation: { ...TEAM1, director_verified_speedup_weighted: 1.5 } });
  const newer = mkTeam('20260910_171000_1_1', { patch: 'x', validation: { ...TEAM1, director_verified_speedup_weighted: 2.5 } });
  const t = Date.now() / 1000 + 60;
  fs.utimesSync(path.join(newer, 'final_patch.diff'), t, t);   // make "newest" unambiguous on a coarse-mtime fs
  const rec = H.recoverLaneFromDisk(EXP, OP, before);
  ok(rec && H.recoveredIso(rec) === 2.5, 'the newest finished team wins when a call produced several');
}

console.log('6. a missing/unreadable exp_root degrades to today\'s behaviour, it never throws');
{
  let threw = false;
  try {
    ok(H.recoverLaneFromDisk(path.join(TMP, 'no_such_root'), OP, new Set()) === null, 'missing exp_root => null');
    ok(H.recoverLaneFromDisk(EXP, 'no_such_op', new Set()) === null, 'unknown op => null');
    ok(H._teamDirs(path.join(TMP, 'no_such_root'), OP).length === 0, '_teamDirs on a missing root => []');
    ok(H.recoveredIso(null) === 0, 'recoveredIso(null) is 0, not a crash');
  } catch (e) { threw = true; console.error('  threw:', e && e.message); }
  ok(!threw, 'no path through the helper throws');
}

console.log('7. opNameOf derives the op from the task dir, with or without a trailing slash');
{
  ok(H.opNameOf(`/a/b/${OP}`) === OP, 'plain path');
  ok(H.opNameOf(`/a/b/${OP}/`) === OP, 'trailing slash');
  ok(H.opNameOf('') === '', 'empty input is empty, not a crash');
  ok(H.opNameOf(null) === '', 'null input is empty, not a crash');
}

console.log('8. every nested-workflow call site is wired to the recovery (structural)');
{
  // The three places a kernel_workflow lane's return can be lost: the serial head author loop, the
  // fast-mode author fan-out, and the milestone kernel track. A new call site added without recovery
  // reopens exactly the fixv5 hole, so pin the count rather than the line numbers.
  // Count real invocations only: skip the definition line and the prose in comments.
  const wired = src.split('\n').filter((l) =>
    /recoverLaneFromDisk\(/.test(l) && !/const recoverLaneFromDisk/.test(l) && !/^\s*\/\//.test(l)).length;
  ok(wired >= 3, `recoverLaneFromDisk is wired into all nested-workflow call sites (found ${wired}, expected >= 3)`);
  ok(/teamSnapshot\(hExpRoot, hOp\)/.test(src), 'serial head author loop takes a before-snapshot');
  ok(/teamSnapshot\(jExpRoot, jOp\)/.test(src), 'fast-mode author fan-out takes a before-snapshot');
  ok(/teamSnapshot\(kExpRoot, kOp\)/.test(src), 'milestone kernel track takes a before-snapshot');
  // The recovery must run BEFORE the retry decision, or it spends a whole team before looking at disk.
  const loop = src.slice(src.indexOf('const AUTHOR_TRIES'), src.indexOf('const alIso = al &&'));
  ok(loop.indexOf('recoverLaneFromDisk(') < loop.indexOf('if (!transient || attempt === AUTHOR_TRIES) break;'),
    'disk recovery is consulted before the "retry with a fresh team" decision');
}

fs.rmSync(TMP, { recursive: true, force: true });
console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
process.exit(failures ? 1 : 0);
