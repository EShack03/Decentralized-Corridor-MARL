"""
[ES] added (17 Nov 2026) - distillation, network update (one DAgger round).

Trains the STUDENT (a copy of the original actor network) by supervised learning on every shard
collected so far (dataset aggregation), then writes a NEW weights folder. The original weights in
--model_dir are only read.

Target = the teacher's whole move distribution: the original policy's distribution restricted to the moves
the filter allows and renormalised, p_T = softmax(o_i masked by the filter). Where the filter removed
nothing this IS the original policy's distribution, so that part of the loss is the "pull back towards the
original policy" of the plan (zero at the start, so the student only moves where the filter had something
to say); where it removed moves the student learns to give them less weight, even when they were not the
policy's choice. Per sample i (student logits z_i, teacher move y_i, w_i = 1 / keep probability):
    KL_i = KL(p_T || softmax(z_i))     weight c_i = w_i * (w_int if the filter changed the move else 1)
    CE_i = -log softmax(z_i)[y_i]      only on seconds where the filter changed the move, weight w_i
    loss = sum(c KL) / sum(c) + ce * sum(w CE) / sum(w)
Hard labels on every second were tried first and rejected (local test, 17 Nov): the original policy is
nearly indifferent between its top moves (mean top probability ~0.3), and fitting hard labels flipped
16-36 % of its choices within two epochs even on the training data, while the loss still fell. The
extra hard-label term (--distill_ce) is off by default: in the 30-40 aircraft local test it cut agreement with
the original where the filter removed nothing from 99 % to 80-89 % (near-equivalent moves) and the student flew
no safer.

The recurrent state fed to the student is the one it had when the data were flown (one-step,
no back-propagation through time) - a standard simplification for recurrent DAgger; noted in the report.

Usage (repo root):
  python onpolicy/scripts/distill_train.py <eval args for any layout, used only to build the network> \
      --distill_data "data/seed1/r*/*.npz" --distill_init <weights dir> --distill_save <new weights dir> \
      [--distill_epochs 3 --distill_lr 3e-5 --distill_w_int 5 --distill_ce 0 --distill_device cpu]
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys
import time

sys.path.append(os.path.abspath(os.getcwd()))

import numpy as np
import torch
import torch.nn.functional as F

from onpolicy.config import get_config
from onpolicy.scripts.eval_mpe import make_render_env, parse_args, modify_args
from onpolicy.scripts.distill_collect import actor_logits

KEYS = ("obs", "node_obs", "agent_id", "rnn", "mask", "label", "orig_logits", "interv", "w", "adj_idx", "fmask", "blocked")


def targs(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--distill_data", type=str, required=True, help="glob of shard files (all rounds so far)")
    p.add_argument("--distill_init", type=str, default="", help="weights folder to start from (empty = --model_dir)")
    p.add_argument("--distill_save", type=str, required=True)
    p.add_argument("--distill_epochs", type=int, default=3)
    p.add_argument("--distill_lr", type=float, default=3e-5)
    p.add_argument("--distill_batch", type=int, default=256)
    p.add_argument("--distill_w_int", type=float, default=5.0)
    p.add_argument("--distill_ce", type=float, default=0.0, help="weight of the hard-label term on intervention seconds")
    p.add_argument("--distill_val_frac", type=float, default=0.05)
    p.add_argument("--distill_device", type=str, default="auto")
    p.add_argument("--distill_max_samples", type=int, default=0, help="cap per shard (0 = all), for quick tests")
    return p.parse_known_args(argv)[0]


def load_shards(pattern, cap, rng):
    """Shards grouped by graph size (node count), so a minibatch always has one node count."""
    groups = {}
    files = sorted(glob.glob(pattern))
    assert files, f"no shards match {pattern}"
    for f in files:
        z = np.load(f)
        if len(z["label"]) == 0:
            continue
        n = int(z["node_obs"].shape[1])     # graph size (aircraft, plus goal nodes on the hidden-goal layouts)
        d = {k: z[k] for k in KEYS}
        d["adj"] = z["adj"]
        if cap and len(d["label"]) > cap:
            keep = np.sort(rng.choice(len(d["label"]), cap, replace=False))
            d = {**{k: d[k][keep] for k in KEYS}, "adj": d["adj"]}
        g = groups.setdefault(n, {k: [] for k in KEYS + ("adj",)})
        off = sum(len(a) for a in g["adj"])
        for k in KEYS:
            g[k].append(d[k] + off if k == "adj_idx" else d[k])
        g["adj"].append(d["adj"])
    return {n: {k: np.concatenate(v) for k, v in g.items()} for n, g in groups.items()}, files


def batches(groups, idx_by_n, bs, rng):
    jobs = []
    for n, idx in idx_by_n.items():
        idx = rng.permutation(idx)
        jobs += [(n, idx[i:i + bs]) for i in range(0, len(idx), bs)]
    rng.shuffle(jobs)
    return jobs


def run_batch(student, g, ix, dev, w_int, ce_w, train):
    t = lambda k, dt=torch.float32: torch.as_tensor(g[k][ix]).to(device=dev, dtype=dt)
    adj = torch.as_tensor(g["adj"][g["adj_idx"][ix]]).to(dev)
    z, _ = actor_logits(student, t("obs"), t("node_obs"), adj, t("agent_id"), t("rnn"), t("mask"))
    y = t("label", torch.long).view(-1)
    interv = t("interv", torch.bool).view(-1)
    w = t("w").view(-1)
    allowed = t("fmask", torch.bool)
    lt = t("orig_logits").masked_fill(~allowed, float("-inf"))
    lpt = torch.log_softmax(lt, -1); pt = lpt.exp()
    lpt = torch.where(allowed, lpt, torch.zeros_like(lpt))          # 0 * log 0 = 0, no NaN
    lz = torch.log_softmax(z, -1)
    kl = (pt * (lpt - lz)).sum(-1)
    c = w * torch.where(interv, torch.full_like(w, w_int), torch.ones_like(w))
    loss = (c * kl).sum() / c.sum()
    wi = w * interv.float()
    if ce_w > 0 and wi.sum() > 0:
        loss = loss + ce_w * (wi * F.cross_entropy(z, y, reduction="none")).sum() / wi.sum()
    pred = z.argmax(-1)
    ok = pred == y
    # near-agreement: the student's move has >= 90 % of the teacher's top probability (the original policy is
    # nearly indifferent between its top moves: median top-1/top-2 logit gap 0.15, so exact agreement is strict)
    near = pt.gather(1, pred[:, None])[:, 0] >= 0.9 * pt.max(-1)[0]
    safe = allowed.gather(1, pred[:, None])[:, 0]          # the student's move is one the filter allows
    blk = t("blocked", torch.bool).view(-1)
    oth = ~interv & ~blk
    out = []
    for m in (interv, blk, oth):
        out += [ok[m].sum().item(), near[m].sum().item(), safe[m].sum().item(), m.sum().item()]
    return loss, out


def main(argv):
    a = targs(argv)
    all_args = parse_args(argv, get_config())
    all_args = modify_args(all_args.model_dir, all_args)
    all_args.use_render = True
    torch.manual_seed(all_args.seed); np.random.seed(all_args.seed)
    rng = np.random.default_rng(all_args.seed)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.distill_device == "auto" else a.distill_device)

    # build the actor exactly as eval does (env only used for the spaces), then copy it as the student
    envs = make_render_env(all_args)
    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
    runner = Runner({"all_args": all_args, "envs": envs, "eval_envs": None, "num_agents": all_args.num_agents,
                     "device": torch.device("cpu"), "run_dir": None})
    student = copy.deepcopy(runner.trainer.policy.actor)
    envs.close()
    init = a.distill_init or all_args.model_dir
    student.load_state_dict(torch.load(os.path.join(init, "actor.pt"), map_location="cpu"))
    student.to(dev); student.tpdv = dict(dtype=torch.float32, device=dev)

    groups, files = load_shards(a.distill_data, a.distill_max_samples, rng)
    tr, va = {}, {}
    for n, g in groups.items():
        p = rng.permutation(len(g["label"])); nv = int(len(p) * a.distill_val_frac)
        va[n], tr[n] = p[:nv], p[nv:]
    n_tr = sum(len(v) for v in tr.values()); n_int = sum(int(groups[n]["interv"][tr[n]].sum()) for n in tr)
    print(f"[distill-train] {len(files)} shards, {n_tr} training samples ({n_int} filter interventions), "
          f"sizes {sorted(groups)}; device {dev}; init {init}", flush=True)

    opt = torch.optim.Adam(student.parameters(), lr=a.distill_lr)
    log = []

    def evaluate():
        """Held-out loss and, on seconds where the filter changed the move / removed other moves only / removed
        nothing: how often the student's move equals the teacher's (agree), is nearly as preferred by the teacher
        (near), and is a move the filter allows (safe; the original policy scores 0 on changed seconds by definition)."""
        student.eval(); acc = np.zeros(12); tot = 0.0; nb = 0
        with torch.no_grad():
            for n, ix in batches(groups, va, 1024, rng):
                l, c = run_batch(student, groups[n], ix, dev, a.distill_w_int, a.distill_ce, False)
                tot += l.item(); nb += 1; acc += c
        r = dict(val_loss=round(tot / max(nb, 1), 5))
        for i, k in enumerate(("changed", "removed_only", "untouched")):
            n = max(acc[4 * i + 3], 1)
            r[f"agree_{k}"] = round(acc[4 * i] / n, 4)
            r[f"near_{k}"] = round(acc[4 * i + 1] / n, 4)
            if k != "untouched":
                r[f"safe_{k}"] = round(acc[4 * i + 2] / n, 4)
        r["n_val_changed"] = int(acc[3])
        return r

    r0 = evaluate(); r0["epoch"] = 0; log.append(r0)
    print(f"[distill-train] before: {r0}", flush=True)
    t0 = time.time()
    for ep in range(1, a.distill_epochs + 1):
        student.train(); tl = 0.0; nb = 0
        for n, ix in batches(groups, tr, a.distill_batch, rng):
            loss = run_batch(student, groups[n], ix, dev, a.distill_w_int, a.distill_ce, True)[0]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            opt.step(); tl += loss.item(); nb += 1
        r = evaluate(); r.update(epoch=ep, train_loss=tl / max(nb, 1), seconds=time.time() - t0); log.append(r)
        print(f"[distill-train] epoch {ep}: {r}", flush=True)

    # new weights folder: a copy of the original folder's network files with actor.pt replaced
    os.makedirs(a.distill_save, exist_ok=True)
    for f in os.listdir(all_args.model_dir):
        if f.endswith((".pt", ".yaml")) and f != "actor.pt":
            shutil.copy2(os.path.join(all_args.model_dir, f), os.path.join(a.distill_save, f))
    student.to("cpu")
    torch.save(student.state_dict(), os.path.join(a.distill_save, "actor.pt"))
    json.dump({"args": vars(a), "shards": files, "log": log}, open(os.path.join(a.distill_save, "distill_log.json"), "w"), indent=1)
    print(f"[distill-train] SAVED {a.distill_save}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
