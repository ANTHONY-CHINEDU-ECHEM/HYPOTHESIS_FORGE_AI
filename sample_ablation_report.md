# HypothesisForge Ablation Report

| Metric | baseline | no_critic | strict_judge | lenient_judge |
|---|---|---|---|---|
| total_hypotheses_considered | 6 | 3 | 6 | 6 |
| acceptance_rate | 0.5 | 1.0 | 0.0 | 0.833 |
| rejection_rate | 0.333 | 0.0 | 0.833 | 0.0 |
| revision_rate | 0.167 | 0.0 | 0.167 | 0.167 |
| mean_novelty_score | 5.852 | 5.603 | 4.275 | 6.323 |
| mean_feasibility_score | 5.482 | 5.0 | 3.063 | 6.573 |
| mean_overall_score | 5.57 | 5.3 | 3.472 | 6.447 |
| mean_contradictions_per_hypothesis | 0.333 | 0.0 | 0.667 | 0.0 |
| iteration_over_iteration_delta | 0.554 | 0.0 | -0.663 | -0.733 |
| final_ranked_mean_score | 7.34 | 5.635 | 0.0 | 7.805 |
| toy_simulation_success_rate | 1.0 | 1.0 | 0.0 | 1.0 |
| wall_clock_seconds | 0.01 | 0.004 | 0.008 | 0.009 |

## Per-iteration score trajectory (mean overall_score)

- **baseline**: [5.293, 5.847]
- **no_critic**: [5.3]
- **strict_judge**: [3.803, 3.14]
- **lenient_judge**: [6.813, 6.08]

## Variant descriptions

- **baseline**: Full pipeline: critic enabled, balanced judge persona.
- **no_critic**: Critic step disabled — every hypothesis auto-accepted.
- **strict_judge**: Critic enabled, strict judge persona.
- **lenient_judge**: Critic enabled, lenient judge persona.

## Interpretation notes

- `no_critic` is expected to show a higher (or unchanged) acceptance rate and zero revision activity, since every hypothesis bypasses feasibility/contradiction screening — compare its `mean_overall_score` and `mean_contradictions_per_hypothesis` against `baseline` to see what the critic step actually catches.
- `strict_judge` vs `lenient_judge` isolates how much of the final score is judge-calibration-dependent rather than hypothesis-quality-dependent — large swings suggest the scoring rubric needs tightening (e.g. more concrete anchor examples in the Critic prompt).
- `iteration_over_iteration_delta` > 0 indicates the revise-and-resubmit loop is measurably improving hypothesis quality across iterations, which is the core claim this ablation is meant to test.