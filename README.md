# RSA Model of AI Debate

This project models how a human judge reasons in AI debate using the **Rational Speech
Act (RSA)** framework. Two AI debaters argue opposing sides; the judge updates their
belief about which side is correct. The goal is to understand whether debate reliably
moves the judge toward the honest answer.

The model builds on **Barnett et al. 2022**, who apply RSA to a single strategic speaker.
We extend it to two advocates with opposing goals.

---

## Quick start

```bash
python main/barnett_rsa.py          # no output — imports only
python main/judge_update_comparison.py   # produces figure3.pdf/png
python main/honesty_check.py             # produces figure4.pdf/png
```

---

## Files

### `barnett_rsa.py` — Core model

Defines the world and all agents. Everything else imports from here.

**World.** A world consists of K=5 sticks, each with a value between 0.1 and 0.9. A world
is **long** (true) if the mean stick value is ≥ 0.5, otherwise **short** (false). There
are 59,049 possible worlds.

**J0 — Literal judge.** Given one revealed stick, what fraction of possible worlds would
be long? This is the baseline — a judge with no model of the speaker's strategy.

**S1 — Strategic speaker.** A speaker with a goal (LONG or SHORT) picks which stick to
reveal. They pick strategically: with high β they strongly favour the stick most likely
to move the judge in their direction. β=0 means random pick; higher β means stronger
cherry-picking.

**J1 — Pragmatic judge (single reveal).** The judge knows the speaker is strategic and
reasons backward: given what was revealed, what does that imply about the world? Takes
a prior over β (how strategic the judge believes the speaker to be).

**J1_two — Pragmatic judge (two advocates).** Extends J1 to debate. Both advocates
reveal a stick; the second advocate cannot pick the same position as the first
(without-replacement). The judge updates on both reveals simultaneously, knowing each
advocate's goal.

---

### `judge_update_comparison.py` — What does the judge believe?

Plots the judge's belief as a function of reveal quality, for consultancy and debate.

**Panel 1 — Consultancy.** A single LONG speaker. Shows the **weak evidence effect**: at
high β, a moderately strong stick actually lowers the judge's belief — the correct
inference is that if a strategic speaker showed only moderate evidence, the world must be
worse. This replicates Barnett's Figure 3.

**Panel 2 — Debate.** LONG vs SHORT. The weak evidence effect disappears. Higher β makes
the judge more decisive (steeper curve) without the non-monotone dip. The debate panel
averages over all valid position pairs and both picking orderings to avoid giving either
debater a protocol advantage.

---

### `honesty_check.py` — Does truth win?

Asks: in worlds where the ground truth is LONG, does the judge end up above 0.5?

For each of the 59,049 worlds, computes the **expected judge belief** — what the judge is
likely to believe after one round, accounting for which sticks the speakers are likely to
reveal (via S1). Only long worlds are shown: the question is whether the judge reaches
truth despite SHORT's deceptive strategy.

**Consultancy panel.** Single LONG speaker. Judge generally trends toward truth but the
weak evidence effect creates outliers at high β — some long worlds get low beliefs.

**Debate panel.** LONG vs SHORT. Judge is more consistently above 0.5; high β helps.
Both the judge's belief surface and the probability of each reveal are symmetrised over
both picking orderings, so neither debater has a systematic first-mover advantage.
