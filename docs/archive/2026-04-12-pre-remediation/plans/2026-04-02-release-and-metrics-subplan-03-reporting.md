# Final Reporting Subplan

1. Join the per-resolution `extended_metrics_table.json` files with the corresponding `summary_*.json` runtime stats.
2. Produce one final table with at least these columns: resolution, model, segm AP, AP50, AP75, bbox AP, precision, recall, F1, train time, train peak memory, inference time, inference peak memory, FPS.
3. Update the all-resolution experiment doc so it carries the final table directly instead of splitting AP numbers from runtime numbers.
4. Keep the wording plain: say whether the numbers were reused or rerun, and why.
5. Spot-check the final markdown table against the source JSON for at least one row per resolution before commit.
