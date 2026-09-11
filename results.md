# RAG Evaluation Report

Generated: 2026-08-11 14:20:14

Test set: `eval/qa_testset.json` (10 questions)

## Summary

| Config | Hit Rate@k | MRR | Avg Keyword Coverage | Avg Latency |
|---|---|---|---|---|
| Baseline (vector-only) | 100% | 1.000 | 100% | 29.91s |
| Full pipeline (hybrid + reranker) | 100% | 1.000 | 100% | 40.81s |

## Baseline (vector-only) — per-question detail

| ID | Category | Coverage | Hit | Latency |
|---|---|---|---|---|
| q1_exact_number_general_window | exact_fact | 100% | True | 104.53s |
| q2_exact_number_electronics | exact_fact | 100% | True | 13.01s |
| q3_paraphrase_refund_timeline | semantic_paraphrase | 100% | True | 18.76s |
| q4_personal_plus_general | personal_plus_general | 100% | True | 21.02s |
| q5_damaged_item_deadline | exact_fact | 100% | True | 60.20s |
| q6_non_refundable_items | list_extraction | 100% | True | 20.62s |
| q7_cod_refund_method | exact_fact | 100% | True | 13.47s |
| q8_escalation_path | exact_fact | 100% | True | 16.38s |
| q9_out_of_scope_should_refuse | refusal | 100% | None | 6.60s |
| q10_multihop_electronics_warranty | multi_hop | 100% | True | 24.46s |

## Full pipeline (hybrid + reranker) — per-question detail

| ID | Category | Coverage | Hit | Latency |
|---|---|---|---|---|
| q1_exact_number_general_window | exact_fact | 100% | True | 260.41s |
| q2_exact_number_electronics | exact_fact | 100% | True | 13.48s |
| q3_paraphrase_refund_timeline | semantic_paraphrase | 100% | True | 17.43s |
| q4_personal_plus_general | personal_plus_general | 100% | True | 18.78s |
| q5_damaged_item_deadline | exact_fact | 100% | True | 12.58s |
| q6_non_refundable_items | list_extraction | 100% | True | 16.11s |
| q7_cod_refund_method | exact_fact | 100% | True | 14.76s |
| q8_escalation_path | exact_fact | 100% | True | 18.10s |
| q9_out_of_scope_should_refuse | refusal | 100% | None | 8.74s |
| q10_multihop_electronics_warranty | multi_hop | 100% | True | 27.66s |