# CLAUDE.md

This file provides guidance for working with the data and prompts from **"Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models"** (Wong, Collins et al., CogSci 2025, arXiv:2507.12547).

## Paper Overview

The paper proposes a **Model Synthesis Architecture (MSA)**: rather than reasoning over all background knowledge at once (intractable), the system constructs a small, bespoke probabilistic program (*M_ad-hoc*) on the fly for each novel situation, then does Bayesian inference inside it.

The MSA is evaluated on a **"Model Olympics"** dataset of sports vignettes (tug-of-war, canoe racing, biathlon) requiring causal inference, with three progressively harder experiments:

- **Exp. 1**: Detailed backgrounds — all causal structure spelled out explicitly.
- **Exp. 2**: Underspecified backgrounds — variable names given, dependencies must be retrieved from world knowledge.
- **Exp. 3**: Participant-generated novel details — arbitrary new variables (injuries, energy drinks) injected as free-text commentary.

Key finding: MSA captures human judgments better than LM-only baselines (direct and chain-of-thought), especially in Exp. 3 (open-world setting). The base LM is `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` via Together API; the PPL is WebPPL with rejection sampling.

## Repository Structure

```
msa-cogsci-2025-data-main/
├── msa-frame-prompts/          # LM prompt templates for each pipeline stage
├── example-scenarios/
│   ├── e1-e2/                  # Examples used for Exp. 1 and Exp. 2
│   └── e3/                     # Extended examples used for Exp. 3
├── lm-only-baseline-prompts/   # Prompts for Direct-LLM and CoT-LLM baselines
└── model-olympics-human-experiment/  # Human participant vignettes and backgrounds
```

## MSA Pipeline

The MSA constructs *M_ad-hoc* in four sequential stages, each with a frame prompt template and an LM-based scoring function:

### Stage 1 — Parse (`generate-parsing.txt`)
Translates each natural language observation (O) and question (Q) into placeholder WebPPL `condition(...)` and query expressions. These reference functions that don't exist yet — they are defined in Stage 4.

**Prompt assembly**: the LLM call combines two files:
- **System message**: `generate-system-prompt.txt` — establishes the LLM as an expert WebPPL programmer with hard syntax constraints (no assignment expressions, no loops, available helper functions, etc.)
- **User/frame message**: `generate-parsing.txt` — few-shot examples from *other* sports (truncated at `<END_LANGUAGE_TO_WEBPPL_CODE>`) + the target scenario

**Scoring**: after generation, `score-parsing.txt` is called as a *separate* LLM prompt with the generated parse injected at `<PARSE_INJECTED_HERE>`. The LLM rates 0–10 whether each sentence's meaning is faithfully preserved. Used to select among candidates when k_parse > 1.

- Temperature: 0.2 (low, syntactic precision needed)
- k_parse = 1 (single sample; scoring via `score-parsing.txt` is available but rarely decisive)
- Delimiter in examples: `<START_LANGUAGE_TO_WEBPPL_CODE>` / `<END_LANGUAGE_TO_WEBPPL_CODE>`

### Stage 2+3 — Informal Background Knowledge + Dependency Graph (`generate-informal-background-knowledge-and-dependency-graph.txt`)
Retrieves relevant causal background (B+) in natural language and simultaneously proposes a conceptual dependency graph G listing all variables and their dependencies.

**Prompt assembly**: same system message (`generate-system-prompt.txt`) + frame message (`generate-informal-background-knowledge-and-dependency-graph.txt`) with examples truncated at `<END_CONCEPT_TRACE>`, and the target scenario + its Stage 1 parse injected at `<SCENARIO_AND_PARSE_INJECTED_HERE>`.

**Scoring**: `score-informal-background-and-dependency-graph.txt` evaluates whether the causal explanation is accurate and whether the concept dependency graph is correctly ordered (concepts defined before dependents). Injected at `<INJECTED_INFORMAL_BACKGROUND_AND_DEPENDENCY_GRAPH_HERE>`.

- Temperature: 0.5 (higher, natural language diversity matters)
- k_informal = 8 candidates, scored via `score-informal-background-and-dependency-graph.txt`; best B*, G* passed forward
- Output delimiters: `<START_SCRATCHPAD>` / `<END_SCRATCHPAD>` and `<START_CONCEPT_TRACE>` / `<END_CONCEPT_TRACE>`

### Stage 4 — Probabilistic Model Synthesis (`generate-model.txt`)
Generates a full WebPPL program M_ad-hoc = (Π_B, Π_O*, Π_Q*). The generated functions must define every placeholder that appeared in Stage 1.

**Prompt assembly**: `generate-model.txt` only — **no system prompt** for this stage. Full examples (all stages concatenated) are injected, plus the target scenario + all prior stage outputs.

- Temperature: 0.2
- k_program = 1; Φ_model is boolean executability (first program that compiles and returns inference results wins)
- Output delimiters: `<START_WEBPPL_MODEL>` / `<END_WEBPPL_MODEL>`

### Stage 5 — Probabilistic Inference
Run the compiled WebPPL program under rejection sampling.

- k_samples = 1000 (Exp. 1/2), 500 (Exp. 3)
- N = 10 simulated participants per vignette

## Frame Prompts and Example Injection

Each frame prompt template contains injection tokens that are filled at runtime:

| Token | Content injected |
|-------|-----------------|
| `<SHUFFLED EXAMPLES ... INJECTED HERE>` | Few-shot examples from *other* sports, truncated to the current stage's delimiter |
| `<SCENARIO_INJECTED_HERE>` | The target vignette (B, O, Q) |
| `<SCENARIO_AND_PARSE_INJECTED_HERE>` | Target vignette + its Stage 1 parse output |

**Held-out prompting scheme**: when processing a tug-of-war vignette, examples are drawn only from canoe-racing, biathlon, diving, and exam — never tug-of-war. This forces generalization.

Each example file (e.g., `example-scenarios/e1-e2/biathlon.txt`) contains **all five stages concatenated** with the above delimiters. Each generation stage slices the example files up to and including the delimiter for that stage, so the LM sees only the context it would have produced so far.

## Example Scenario File Format

```
<START_SCENARIO>
BACKGROUND: ...
CONDITIONS: ...
QUERIES: ...
<END_SCENARIO>

<START_LANGUAGE_TO_WEBPPL_CODE>
// condition and query expressions
<END_LANGUAGE_TO_WEBPPL_CODE>

<START_SCRATCHPAD>
// informal NL background knowledge B+
<END_SCRATCHPAD>

<START_CONCEPT_TRACE>
// dependency graph G
<END_CONCEPT_TRACE>

<START_WEBPPL_MODEL>
// full WebPPL probabilistic program
<END_WEBPPL_MODEL>
```

## Example Scenario Domains

| Domain | Used in | Notes |
|--------|---------|-------|
| tug-of-war | Exp. 1/2/3 | Replicates Goodman et al. 2014; constant strength + per-match effort |
| canoe-racing | Exp. 1/2/3 | Novel domain; same causal structure as tug-of-war |
| biathlon | Exp. 1/2 | More distinct; strength + shooting accuracy |
| diving | Examples only | Synchronised diving; used as a general example |
| exam | Examples only (e1-e2 only) | Student performance; used as a general example |

## WebPPL Patterns Used in Models

- `mem(function({athlete}) { ... })` — memoize stochastic attributes that are constant per entity
- `any_previous_time_inclusive(fn, t)` — helper for temporal effects (e.g., injury in a prior match)
- `Infer({ model: model, method: 'rejection' })` — rejection sampling inference
- Team aggregation via `mean(map(...))` or `sum(map(...))`
- Match outcomes: `team1_score > team2_score`
- Ranking queries: count how many of N random athletes score lower than the target athlete

## Human Experiment Details

- Multi-click interface: participants give k_click = 5 judgments per question (treated as samples from their posterior)
- 8 questions per vignette: 3 about constant latent variable, 3 about temporally varying latent, 2 match predictions
- Comparison metrics: R² (correlational) and Wasserstein Distance (distributional)
- Split-half human-human baseline used as noise ceiling
