# Code Benchmark Notes

This mean@32 suite is a prompt-generation evaluator. It can score tasks whose final
answer can be matched from text, such as AIME, SciBench, and AMO-Bench.

LiveCodeBench, Codeforces, and SWE-bench require executable evaluation:

- LiveCodeBench and Codeforces need generated code to be run against hidden or
  provided tests, then scored with pass@k.
- SWE-bench needs repository checkout, patch application, dependency setup, and
  project tests inside a sandbox.

Those benchmarks should be run with their official evaluators or a dedicated
execution harness. Adding them here as plain `generate_until` tasks would only
save model responses and would not produce a trustworthy benchmark score.
