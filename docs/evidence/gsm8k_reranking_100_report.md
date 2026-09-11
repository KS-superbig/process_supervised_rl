# final-only vs final+process candidate selection report

## Summary
- samples: 100
- candidates: 400
- final_only_accuracy: 0.9300
- final_plus_process_accuracy: 0.9300
- final_only_selected_process_reward_mean: 0.6681
- final_plus_selected_process_reward_mean: 0.6978
- changed_selection_count: 50
- changed_selection_rate: 0.5000
- all_candidate_process_reward_mean: 0.6365
- corr(num_steps, process_reward): -0.8478

## Flags
- low_progress_signal: 4

## Changed Top1 Cases
- gsm8k-train-000003: gsm8k-train-000003-cand-02 -> gsm8k-train-000003-cand-03; final=1.0->1.0; process=0.6857->0.7047
- gsm8k-train-000004: gsm8k-train-000004-cand-02 -> gsm8k-train-000004-cand-03; final=1.0->1.0; process=0.6924->0.7030
- gsm8k-train-000006: gsm8k-train-000006-cand-01 -> gsm8k-train-000006-cand-04; final=0.0->0.0; process=0.5513->0.6373
- gsm8k-train-000007: gsm8k-train-000007-cand-01 -> gsm8k-train-000007-cand-02; final=1.0->1.0; process=0.7257->0.7548
- gsm8k-train-000009: gsm8k-train-000009-cand-01 -> gsm8k-train-000009-cand-04; final=1.0->1.0; process=0.6845->0.7459
- gsm8k-train-000011: gsm8k-train-000011-cand-01 -> gsm8k-train-000011-cand-03; final=1.0->1.0; process=0.6333->0.6355
- gsm8k-train-000016: gsm8k-train-000016-cand-01 -> gsm8k-train-000016-cand-02; final=0.0->0.0; process=0.5285->0.7714
- gsm8k-train-000021: gsm8k-train-000021-cand-01 -> gsm8k-train-000021-cand-04; final=1.0->1.0; process=0.7314->0.7543
- gsm8k-train-000022: gsm8k-train-000022-cand-01 -> gsm8k-train-000022-cand-03; final=1.0->1.0; process=0.6267->0.6586
- gsm8k-train-000027: gsm8k-train-000027-cand-02 -> gsm8k-train-000027-cand-03; final=1.0->1.0; process=0.5952->0.6957
- gsm8k-train-000028: gsm8k-train-000028-cand-01 -> gsm8k-train-000028-cand-04; final=1.0->1.0; process=0.6696->0.6998
- gsm8k-train-000029: gsm8k-train-000029-cand-01 -> gsm8k-train-000029-cand-02; final=1.0->1.0; process=0.7286->0.7365
- gsm8k-train-000031: gsm8k-train-000031-cand-01 -> gsm8k-train-000031-cand-02; final=1.0->1.0; process=0.6455->0.6852
- gsm8k-train-000032: gsm8k-train-000032-cand-02 -> gsm8k-train-000032-cand-04; final=1.0->1.0; process=0.6409->0.7158
- gsm8k-train-000033: gsm8k-train-000033-cand-01 -> gsm8k-train-000033-cand-02; final=1.0->1.0; process=0.6762->0.7238
- gsm8k-train-000038: gsm8k-train-000038-cand-01 -> gsm8k-train-000038-cand-02; final=1.0->1.0; process=0.6727->0.6926
- gsm8k-train-000039: gsm8k-train-000039-cand-01 -> gsm8k-train-000039-cand-02; final=1.0->1.0; process=0.7076->0.7175
- gsm8k-train-000042: gsm8k-train-000042-cand-01 -> gsm8k-train-000042-cand-04; final=1.0->1.0; process=0.6711->0.7238
- gsm8k-train-000044: gsm8k-train-000044-cand-01 -> gsm8k-train-000044-cand-04; final=1.0->1.0; process=0.6577->0.6833
- gsm8k-train-000048: gsm8k-train-000048-cand-01 -> gsm8k-train-000048-cand-04; final=1.0->1.0; process=0.5762->0.6846

## First-pass Reading Guide
- If final_plus_process_accuracy is lower than final_only_accuracy, process reward is hurting selection.
- If accuracy is tied and selected process reward is higher, process reward is adding useful ranking signal.
- Changed cases should be manually inspected before moving to training.

## Additional Checks
- all_candidate_final_accuracy: 0.6150
- samples_with_at_least_one_correct_candidate: 93 / 100
- changed_selection_breakdown: 1->1 = 46, 0->0 = 4, 1->0 = 0, 0->1 = 0
- unchanged_selection_breakdown: same_correct = 47, same_wrong = 3
- tokenizer_artifact_rows_after_cleaning: 0

## First-pass Conclusion
- final+process did not reduce top1 final accuracy: both strategies selected correct answers for 93 / 100 samples.
- final+process selected candidates with higher mean process reward: 0.6978 vs 0.6681.
- process reward changed top1 selection for 50 / 100 samples, mostly among candidates with the same final correctness.
- This supports process reward as a useful reranking signal, but the strong negative correlation with step count (-0.8478) still needs manual review before using it for training.

