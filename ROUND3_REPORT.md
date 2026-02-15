# Local LLM Tool-Calling Benchmark Report (Round 3)

**Date:** 2026-02-14
**Models:** 21 (qwen3:4b removed for impractical latency; nanbeige4.1:3b added)
**Runs:** 20 per model/prompt combination (4,800 total inference calls for the original 20 models; nanbeige4.1:3b ran 3 times only)
**Hardware:** CPU-only (no GPU acceleration)

## What Changed from Round 2

Round 3 uses the same prompts, scoring formula, parsers, and backends as Round 2. Two things changed:

1. **20 runs per model** (up from 3). This is the entire point of Round 3. The Round 2 results showed that 3-run majority voting has high variance on borderline prompts -- bitnet-2B-4T shifted by 0.240 between Round 1 and Round 2 reruns. An order-dependence experiment on lfm2.5:1.2b confirmed that the variance comes from stochastic sampling at temperature 0.7, not from prompt ordering. With 20 runs, majority voting stabilizes and reveals the true decision boundaries.

2. **qwen3:4b removed.** At 63.7 seconds per prompt, running 20 iterations would take ~4.25 hours for a single model. It tied with qwen3:0.6b in Round 2 (both 0.880) with identical majority-vote behavior but 17x the latency. Removing it cut total benchmark time roughly in half.

3. **nanbeige4.1:3b added.** A community-requested Chinese reasoning model (Nanbeige Lab). Not available in the official Ollama library, so it runs via llama.cpp with a Q4_K_M GGUF quantisation. Due to extremely high CPU latency (~23s per prompt), it was run 3 times instead of 20. Its scores should be treated as preliminary.

Total inference: 20 models x 12 prompts x 20 runs + 1 model x 12 prompts x 3 runs = 4,836 calls. Runtime: ~5.3 hours on CPU.

## Machine Specs

Unchanged from Round 2.

| Component | Detail |
|---|---|
| CPU | AMD Ryzen AI 7 350 w/ Radeon 860M |
| Cores / Threads | 8 cores / 16 threads |
| RAM | 32 GB DDR5 |
| GPU | None used |
| OS | Arch Linux, kernel 6.18.3-arch1-1 |

## Results

### Full Leaderboard (sorted by Agent Score)

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Reliability | Multi-Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **qwen3:1.7b** | Ollama | native-tools | CN | 0.900 | 1.000 | 0 | 0.800 | N/A* | **0.960** | 10,665 |
| **2** | **lfm2.5:1.2b** | llama.cpp | openai-compat | US | 0.800 | 1.000 | 0 | 0.725 | 1.000 | **0.920** | 1,567 |
| 3 | qwen3:0.6b | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | 0.729 | N/A* | 0.880 | 3,410 |
| 4 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.500 | 1.000 | 0 | 0.625 | N/A* | 0.800 | 2,240 |
| 4 | ministral-3:3b | Ollama | native-tools | FR | 0.500 | 1.000 | 0 | 0.583 | N/A* | 0.800 | 7,505 |
| 4 | nanbeige4.1:3b† | llama.cpp | openai-compat | CN | 0.500 | 1.000 | 0 | 0.583 | 1.000 | 0.800 | 22,812 |
| **7** | **phi4-mini:3.8b** | Ollama | raw-schema | US | 0.700 | 1.000 | 1 | 0.683 | 1.000 | **0.780** | 5,180 |
| **8** | **gemma3:1b** | Ollama | raw-schema | US | 0.600 | 0.500 | 0 | 0.533 | 0.000 | **0.690** | 2,321 |
| 9 | qwen2.5:3b | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.700 | N/A* | 0.670 | 3,014 |
| 10 | llama3.2:3b | Ollama | native-tools | US | 0.900 | 0.000 | 0 | 0.750 | N/A* | 0.660 | 1,690 |
| 11 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.675 | N/A* | 0.640 | 1,015 |
| 11 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.683 | N/A* | 0.640 | 1,722 |
| 11 | functiongemma | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.667 | N/A* | 0.640 | 435 |
| 14 | smollm3:3b | Ollama | raw-schema | US | 0.700 | 0.500 | 1 | 0.667 | 1.000 | 0.630 | 9,727 |
| 15 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.296 | 0.000 | 0.600 | 1,549 |
| 15 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 | 14,157 |
| **17** | **jan-v3:4b** | Ollama | raw-schema | US | 0.900 | 0.000 | 1 | 0.750 | 0.500 | **0.560** | 2,322 |
| **18** | **bitnet-2B-4T** | bitnet.cpp | openai-compat | US/1bit | 0.700 | 0.500 | 2 | 0.721 | 1.000 | **0.530** | 2,075 |
| 19 | granite3.3:2b | Ollama | native-tools | US | 0.800 | 0.000 | 1 | 0.571 | N/A* | 0.520 | 1,658 |
| 19 | granite4:3b | Ollama | native-tools | US | 0.800 | 0.000 | 1 | 0.696 | N/A* | 0.520 | 2,112 |
| 21 | llama3.2:1b | Ollama | native-tools | US | 0.700 | 0.500 | 3 | 0.667 | N/A* | 0.430 | 1,596 |

\*Ollama native-tools API returns only the first tool call. **Bold** = significant ranking change from Round 2. †nanbeige4.1:3b was run 3 times (not 20) due to high CPU latency; its scores should be treated as preliminary.

### Edge Agent Mini Leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Action | Restraint | Wrong Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|
| 1 | **qwen3:1.7b** | Ollama | native-tools | 0.900 | 1.000 | 0 | **0.960** | 10,665 |
| 2 | **lfm2.5:1.2b** | llama.cpp | openai-compat | 0.800 | 1.000 | 0 | **0.920** | 1,567 |
| 3 | qwen3:0.6b | Ollama | native-tools | 0.700 | 1.000 | 0 | 0.880 | 3,410 |
| 4 | qwen2.5:1.5b | Ollama | native-tools | 0.500 | 1.000 | 0 | 0.800 | 2,240 |
| 5 | gemma3:1b | Ollama | raw-schema | 0.600 | 0.500 | 0 | 0.690 | 2,321 |
| 6 | qwen2.5:0.5b | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.640 | 1,015 |
| 6 | smollm2:1.7b | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.640 | 1,722 |
| 6 | functiongemma | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.640 | 435 |
| 9 | deepseek-r1:1.5b | Ollama | raw-schema | 0.000 | 1.000 | 0 | 0.600 | 1,549 |
| 10 | bitnet-2B-4T | bitnet.cpp | openai-compat | 0.700 | 0.500 | 2 | 0.530 | 2,075 |
| 11 | llama3.2:1b | Ollama | native-tools | 0.700 | 0.500 | 3 | 0.430 | 1,596 |

### Hard Prompts P10-P12 (detailed)

| Model | P10 Tool | P10 | P11 Tool | P11 | P12 Tool | P12 | Wrong |
|---|---|---|---|---|---|---|---|
| qwen3:1.7b | get_weather | OK | search_files | OK | schedule_meeting | OK | 0 |
| lfm2.5:1.2b | get_weather | OK | (none) | miss | (none) | miss | 0 |
| qwen3:0.6b | (none) | miss | search_files | OK | (none) | miss | 0 |
| qwen2.5:1.5b | (none) | miss | (none) | miss | schedule_meeting | OK | 0 |
| ministral-3:3b | (none) | miss | (none) | miss | (none) | miss | 0 |
| phi4-mini:3.8b | get_weather | OK | (none) | miss | get_weather | WRONG | 1 |
| gemma3:1b | search_files | wrong? | search_files | OK | search_files | wrong? | 0 |
| qwen2.5:3b | get_weather | OK | search_files | OK | get_weather | WRONG | 1 |
| llama3.2:3b | search_files | wrong? | search_files | OK | schedule_meeting | OK | 0 |
| qwen2.5:0.5b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| smollm2:1.7b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| functiongemma | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| smollm3:3b | (none) | miss | (none) | miss | get_weather | WRONG | 1 |
| deepseek-r1:1.5b | (none) | miss | (none) | miss | (none) | miss | 0 |
| bitnet-3B | (none) | miss | (none) | miss | (none) | miss | 0 |
| jan-v3:4b | get_weather | OK | search_files | OK | get_weather | WRONG | 1 |
| bitnet-2B-4T | schedule_meeting | WRONG | (none) | miss | get_weather | WRONG | 2 |
| granite3.3:2b | get_weather | OK | (none) | miss | get_weather | WRONG | 1 |
| granite4:3b | get_weather | OK | search_files | OK | get_weather | WRONG | 1 |
| llama3.2:1b | schedule_meeting | WRONG | get_weather | WRONG | get_weather | WRONG | 3 |
| nanbeige4.1:3b† | (none) | miss | (none) | miss | (none) | miss | 0 |

†3 runs only. **Only qwen3:1.7b gets all three hard prompts right.** It is the only model in the benchmark to call the correct tool on P10, P11, and P12 simultaneously across 20-run majority voting.

## Score Changes from Round 2

| Model | Round 2 (3-run) | Round 3 (20-run) | Change | What Changed |
|---|---|---|---|---|
| **qwen3:1.7b** | 0.670 | **0.960** | **+0.290** | P5/P9 restraint now passes (was failing in 2/3 runs). All three hard prompts correct. The biggest winner. |
| **lfm2.5:1.2b** | 0.880 | **0.920** | **+0.040** | P10 now passes by majority vote (was a miss). Stable on everything else. |
| qwen3:0.6b | 0.880 | 0.880 | 0.000 | Rock-solid. Same score at 3 runs and 20 runs. |
| qwen2.5:1.5b | 0.840 | 0.800 | -0.040 | P4 and P7 now miss (unreliable at 20 runs). |
| ministral-3:3b | 0.800 | 0.800 | 0.000 | Perfectly stable. |
| **phi4-mini:3.8b** | 0.880 | **0.780** | **-0.100** | P12 now majority-votes to get_weather (WRONG). P6 now misses. Former co-champion drops to 6th. |
| **gemma3:1b** | 0.550 | **0.690** | **+0.140** | P2, P3, P7, P10, P11 now pass. Zero wrong tools. The 3-run sample was deeply unfair. |
| qwen2.5:3b | 0.670 | 0.670 | 0.000 | Stable. |
| llama3.2:3b | 0.660 | 0.660 | 0.000 | Stable. |
| **qwen2.5:0.5b** | 0.640 | 0.640 | 0.000 | Stable. |
| **smollm2:1.7b** | 0.740 | **0.640** | **-0.100** | P11 now majority-votes to get_weather (WRONG). Was lucky at 3 runs. |
| functiongemma | 0.640 | 0.640 | 0.000 | Stable. |
| **smollm3:3b** | 0.710 | **0.630** | **-0.080** | P10 and P11 now miss (were passing at 3 runs). P5 restraint failure confirmed. |
| deepseek-r1:1.5b | 0.720 | 0.600 | -0.120 | Action collapsed to 0.000. Tool calling was a fluke at 3 runs. |
| bitnet-3B | 0.600 | 0.600 | 0.000 | Stable (never calls tools). |
| jan-v3:4b | 0.560 | 0.560 | 0.000 | Stable. |
| **bitnet-2B-4T** | 0.810 | **0.530** | **-0.280** | P10 now WRONG (schedule_meeting), P12 now WRONG (get_weather). The biggest loser -- its Round 2 score was almost entirely sampling luck. |
| **granite3.3:2b** | 0.480 | **0.520** | **+0.040** | Slight improvement. |
| **granite4:3b** | 0.670 | **0.520** | **-0.150** | P2 now misses, P5 and P9 restraint both fail. Was overrated at 3 runs. |
| llama3.2:1b | 0.430 | 0.430 | 0.000 | Stable at the bottom. |

### The Biggest Movers

**qwen3:1.7b: 11th to 1st (+0.290).** The "middle child" from Round 2 turns out to be the benchmark champion. At 3 runs, it happened to fail P5 and P9 restraint in 2/3 runs each, producing a Restraint score of 0.500. At 20 runs, it passes both with comfortable margins. It's also the only model to get all three hard prompts right (P10: get_weather, P11: search_files, P12: schedule_meeting). Agent Score 0.960 is the highest in the benchmark's history.

**bitnet-2B-4T: 6th to 17th (-0.280).** The opposite story. Its Round 2 score of 0.810 reflected lucky sampling on P10 and P12 -- both hard prompts where it makes the wrong call more often than not. At 20 runs, P10 majority-votes to schedule_meeting (WRONG) and P12 to get_weather (WRONG), earning 2 wrong-tool penalties that destroy its score. Its P5 restraint also fails consistently. The model has strong tool-call mechanics but poor judgment.

**phi4-mini:3.8b: tied-1st to 6th (-0.100).** The former co-champion's P12 weakness was hidden by 3-run variance. At 20 runs, it majority-votes to get_weather on P12 (70% of runs), which is the penalized wrong tool. Still has perfect restraint and good P10 performance, but one wrong tool costs 10% of the Agent Score.

**gemma3:1b: 19th to 7th (+0.140).** Five prompts that were misses at 3 runs now pass at 20 runs. Its Action score jumped from 0.500 to 0.600, and crucially, zero wrong tools. The 3-run sample was heavily biased against this model.

## What 20 Runs Revealed

### 3-run majority voting was inadequate for this benchmark

The Round 2 leaderboard had a four-way tie at 0.880. At 20 runs, those four models score 0.960, 0.920, 0.880, and 0.780 -- spread across a 0.180 range. The tie was a sampling artifact.

Seven models changed by more than 0.05 Agent Score. The direction was unpredictable: some went up (qwen3:1.7b, gemma3:1b), others went down (phi4-mini, bitnet-2B-4T, granite4). Without knowing which way a model's luck went at 3 runs, you can't know which direction 20 runs will push it.

The models that were stable across both sample sizes (qwen3:0.6b, ministral-3:3b, qwen2.5:3b, llama3.2:3b, functiongemma, bitnet-3B, jan-v3:4b, llama3.2:1b) share a pattern: they're either strongly above or strongly below the decision threshold on each prompt. Instability concentrates on models that are borderline -- right around 50% call rate on specific prompts.

### The Qwen3 family ranking flipped

Round 2: qwen3:0.6b (0.880) > qwen3:4b (0.880) > qwen3:1.7b (0.670). The 1.7B was the "middle child."

Round 3: **qwen3:1.7b (0.960) > qwen3:0.6b (0.880)**. The 1.7B is now the undisputed champion. Its Round 2 restraint failures were borderline calls that happened to go wrong in 2/3 runs. At 20 runs, it passes P5 and P9 comfortably. It's also the only model that nails all three hard prompts.

The Qwen3 scaling story is now monotonic again: 1.7B > 0.6B. Larger is better within this family, as you'd expect -- the Round 2 non-monotonic ranking was a small-sample artifact.

### Reliability as a predictor

The Reliability metric (average per-prompt success rate before majority voting) now has 20 data points per prompt instead of 3. Models with Reliability > 0.70 tend to be stable across sample sizes. Models with Reliability < 0.60 are the ones whose scores shifted most.

| Model | Reliability | Score Change | Stable? |
|---|---|---|---|
| qwen3:1.7b | 0.800 | +0.290 | No (borderline prompts flipped) |
| lfm2.5:1.2b | 0.725 | +0.040 | Yes |
| qwen3:0.6b | 0.729 | 0.000 | Yes |
| bitnet-2B-4T | 0.721 | -0.280 | No (borderline prompts flipped) |

Reliability alone doesn't predict stability -- it depends on *which* prompts are borderline. qwen3:1.7b has high reliability but its borderline prompts happened to be the restraint prompts (P5, P9), which carry 30% of the Agent Score.

### The conservative strategy still wins -- but not as dominantly

The top 3 models all have perfect restraint and zero wrong tools, confirming the Round 2 finding. But the new #1 (qwen3:1.7b) has Action 0.900 -- the highest among the top tier. This model doesn't win by being conservative; it wins by being both aggressive *and* accurate. It calls tools on 9 of 10 actionable prompts while maintaining perfect discipline on restraint prompts.

This is a qualitative shift. In Round 2, the winners were conservative (Action 0.700, Restraint 1.000). In Round 3, the winner is the model that can do both: high action AND perfect restraint. The "just decline everything uncertain" strategy still works (ministral at 0.800 with Action 0.500), but it's no longer the best.

## Conclusions

1. **3 runs was not enough.** The Round 2 leaderboard was substantially wrong on 7 of 21 models. The four-way tie at 0.880 was a sampling artifact. With 20 runs, qwen3:1.7b separates clearly as the #1 model (0.960), and the true ranking spread is 0.180 wider than it appeared. Any tool-calling benchmark using small-sample majority voting should report confidence intervals or use more runs.

2. **qwen3:1.7b is the benchmark champion.** The only model to achieve perfect restraint, zero wrong tools, AND Action 0.900 simultaneously. The only model to get all three hard prompts right. Its Round 2 ranking (11th) was a statistical accident -- borderline P5/P9 calls that happened to fail in 2/3 runs.

3. **lfm2.5:1.2b confirms as a top-tier model.** 0.920 Agent Score, 1,567 ms latency, perfect restraint. Now also passes P10 (implicit weather reasoning). Still the best speed/quality ratio in the benchmark by a wide margin -- nearly 7x faster than qwen3:1.7b at a cost of 0.040 Agent Score.

4. **Models that looked great at 3 runs may have been lucky.** bitnet-2B-4T (-0.280), phi4-mini (-0.100), and granite4 (-0.150) all dropped significantly. Their Round 2 scores reflected favorable sampling on borderline hard prompts. The reverse is also true: gemma3:1b (+0.140) and qwen3:1.7b (+0.290) were unlucky at 3 runs.

5. **deepseek-r1:1.5b effectively doesn't work for tool calling.** Action 0.000 at 20 runs means it never reliably produces a parseable tool call by majority vote. Its Round 2 score of 0.720 came from a handful of lucky parses. The model's thinking traces show tool-calling *intent* but its output format is too inconsistent to be useful.

6. **P12 remains the hardest prompt** -- but now we have a model that solves it. qwen3:1.7b correctly calls schedule_meeting on P12 in the majority of runs. Only 3 other models even attempt the right tool (qwen2.5:1.5b, llama3.2:3b, and -- newly visible at 20 runs -- none others). Eleven models call get_weather (the penalized wrong tool).

7. **The gap between "works" and "works reliably" is wider than expected.** Many models hover around 50% call rate on borderline prompts. At 3 runs, a coin flip decides their score. At 20 runs, the truth emerges. This has implications for deployment: a model that passes a prompt 55% of the time will appear to "work" in testing but fail often in production.
