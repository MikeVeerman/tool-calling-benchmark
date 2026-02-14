# Local Agent Bench

**Can a $1,000 laptop run an AI agent that knows when to use tools -- and when not to?**

I tested 20 small open-weight models locally on CPU to see which ones can act -- and which ones know when not to. No cloud API. No GPU. Just Ollama, a handful of 1-bit and 4-bit quantised models, and a Framework 13 running Arch Linux.

[Round 1](ROUND1_REPORT.md) tested 11 models from 7 organisations. After the post [went viral on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1qyg10z/), [Round 2](ROUND2_REPORT.md) added 10 community-requested models -- including every model that was suggested in the comments. [Round 3](ROUND3_REPORT.md) reran all 20 models with 20 runs each (up from 3) to stabilize the rankings and eliminate small-sample variance.

The motivation is practical. Local and private AI agents are increasingly attractive -- no per-token costs, no data leaving the machine, no vendor lock-in. But an agent that acts incorrectly is worse than one that does nothing: a wrong API call costs money, sends the wrong message, or deletes the wrong file. The hard problem isn't generating well-formed JSON. It's deciding whether to act at all.

This benchmark measures **judgment** -- whether a model knows *when* to call a tool -- not just **execution** -- whether it can format a tool call correctly.

## TL;DR

- **qwen3:1.7b is the benchmark champion** at 0.960 Agent Score -- the only model to get all three hard prompts right while maintaining perfect restraint. It was ranked 11th in Round 2 with only 3 runs; 20 runs revealed it was always this good.
- **lfm2.5:1.2b (0.920) is the best speed/quality ratio** -- a 1.2B state-space hybrid at 1,567 ms, nearly 7x faster than qwen3:1.7b for 0.040 less Agent Score.
- **The Round 2 four-way tie at 0.880 was a small-sample artifact.** At 20 runs, those four models score 0.960, 0.920, 0.880, and 0.780. Seven models changed by more than 0.05.
- **3-run majority voting was inadequate.** bitnet-2B-4T dropped from 0.810 to 0.530, phi4-mini from 0.880 to 0.780, while qwen3:1.7b rose from 0.670 to 0.960. Small samples hide the truth on borderline prompts.
- Every model that successfully emits tool calls can handle simple, unambiguous tool calls on CPU at 1-8s latency.
- When prompts require judgment -- resisting keyword triggers, respecting negation, noticing redundant information -- most sub-4B models fail.
- The #1 model wins by being both aggressive *and* accurate (Action 0.900 + Restraint 1.000), not just by being conservative.
- Parameter count is a weak predictor. A 600M model (qwen3:0.6b) scores 0.880 while a 3.8B model (phi4-mini) scores 0.780.
- P12 ("The weather is 8°C and rainy. Should I schedule a meeting?") remains the hardest prompt. Only qwen3:1.7b gets it right reliably at 20 runs.
- **Format compliance masks true behavior -- in both directions.** Parser fixes for 5 models revealed that format-blind scoring both underestimates and overestimates models.

## Why this exists

Tool-calling is the backbone of AI agents. An LLM that can reliably decide "this prompt needs `get_weather`, that one needs `schedule_meeting`, and this other one needs *nothing at all*" is the difference between a useful agent and an expensive autocomplete.

But there's a harder question: when a prompt mentions "weather" but the correct action is *not* to call `get_weather`, can the model resist the keyword trigger? When the user says "don't check the weather, just find the report," does the model listen? When the weather is already provided in the prompt, does the model notice?

Cloud models handle this well. But what about local models running on your laptop's CPU? The small open-weight models (0.5B-3.8B parameters) that Ollama makes trivially easy to run -- can they actually *do* this?

This benchmark tests all of that: 20 models from 10 organisations across 4 countries, 12 prompts, 20 runs each, on a machine with no discrete GPU.

## The test machine

| Spec | Value |
|---|---|
| Laptop | Framework Laptop 13 (AMD Ryzen AI 300 Series) |
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 threads @ 2.0 GHz |
| RAM | 32 GB DDR5 |
| GPU | None used (Radeon 860M iGPU present but not utilised) |
| OS | Arch Linux (kernel 6.18.3) |
| Ollama | v0.13.5 |

Everything runs on CPU. This is intentional -- the point is to test what's achievable on hardware most developers already own.

## The models and why they were chosen

### Round 1: the original 11

**Qwen 2.5 (3B, 1.5B, 0.5B) -- the scaling ladder.** Alibaba's Qwen 2.5 is one of the strongest open model families for tool-calling at small sizes. Testing all three sizes gives a clean read on how capability scales with parameters.

**LLaMA 3.2:3B -- Meta's contender.** The obvious comparison point. Native tool-calling support in Ollama, widely used, the model most people would reach for first.

**SmolLM2:1.7B -- the underdog.** HuggingFace's purpose-built small model. At 1.7B parameters it sits between Qwen's 1.5B and 3B. Tests whether the "small model" space has dark horses.

**Ministral-3:3B -- the EU sovereignty candidate.** Mistral's 3B edge model, Apache 2.0 licensed. The model you'd pick for European-sourced tool-calling.

**DeepSeek-R1:1.5B -- the reasoning distillation.** DeepSeek's distilled chain-of-thought model. Does thinking before answering improve restraint or just burn tokens?

**Gemma3:1B -- Google's smallest.** Sliding window attention architecture at 1B parameters. Tests the floor for tool-calling capability.

**Phi4-mini:3.8B -- Microsoft's reasoning model.** Slightly larger than the 3B tier but trained specifically for structured reasoning. Tests whether Microsoft's approach translates to tool-calling.

**BitNet b1.58-3B -- the 1-bit base model.** Microsoft's 1.58-bit quantisation ({-1, 0, 1} ternary weights). A base model without instruction tuning, included as a control.

**BitNet b1.58-2B-4T -- the 1-bit instruction-tuned model.** Same ternary architecture, instruction-tuned on 4 trillion tokens. Answers the question: can ternary weights produce structured output?

### Round 2: community-requested models

After the [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1qyg10z/), the community requested specific models. Every viable suggestion was added.

**Qwen 3 (1.7B, 0.6B) -- the most-requested family.** Six separate users asked for Qwen3. Both sizes have built-in thinking capability. The 0.6B is the smallest model in the benchmark with native tool support. Tests whether the Qwen2.5 → Qwen3 generation jump matters for tool calling. (The 4B was tested in Round 2 but removed in Round 3 due to impractical latency -- 63s per prompt on CPU.)

**FunctionGemma (270M) -- the specialist.** A 270M fine-tune built specifically for function calling. Two users predicted "very high performance per compute." At 270M it's the smallest model in the benchmark. Tests whether purpose-built fine-tuning beats general instruction tuning.

**Granite 3.3:2B and Granite 4:3B -- IBM's generational test.** One user said Granite4 "just felt good" for tool calling. Including both generations tests whether IBM's model improvements translate to measurable gains on the same benchmark.

**LLaMA 3.2:1B -- Meta's smallest.** The 1B sibling of the Round 1 LLaMA 3.2:3B. Tests how far Meta's tool-calling training extends down the size ladder.

**LFM 2.5:1.2B (Liquid AI) -- the architectural outlier.** A state-space hybrid model, not a transformer. Three users recommended it, with one calling it "a fantastic job for its size." Required a new llama.cpp backend since it's not available through Ollama. Tests whether non-transformer architectures can do tool calling.

**SmolLM3:3B -- the successor.** HuggingFace's follow-up to SmolLM2 with thinking capability. Not yet in Ollama's official library (pulled from HuggingFace GGUF). Tests generational improvement within HuggingFace's small model line.

**Jan v3:4B (jan.ai) -- the fine-tune.** A Qwen3-based fine-tune recommended by two users. Tests whether community fine-tuning on top of Qwen3 improves tool-calling behaviour.

## The prompts

The benchmark uses 12 prompts that escalate in difficulty:

**Easy (P1-P3):** Direct tool calls. "What's the weather in Antwerp?" should obviously call `get_weather`. These establish whether a model can do the basics.

**Ambiguous (P4):** "I'm heading to Brussels tomorrow, anything I should know?" -- calling `get_weather` is reasonable but not required. This tests whether models make sensible judgment calls.

**Restraint (P5, P9):** Prompts where the *correct* answer is to NOT call a tool. P5 asks "What tools do you have access to?" (a meta question). P9 asks "Can you write a Python script that checks the weather using an API?" (a code-writing request that mentions "weather" as a keyword trap). These are the most interesting tests -- an agent that calls tools when it shouldn't is worse than one that occasionally misses a valid call.

**Hard (P6-P8):** P6 requires context the model doesn't have ("the city where we have our next sprint review"). P7 buries meeting parameters in messy natural language with filler words. P8 asks for two tools at once ("search files AND tell me the weather") to see if models handle multi-tool requests or just pick one.

**Hard -- judgment traps (P10-P12):** The hardest prompts, added in Round 4 to break the Round 3 plateau where four models tied at 0.929. These test whether models can pick the *right* tool when misleading keywords are present:

- **P10:** "I have a meeting with a client in Bruges next Thursday. Should I take the train or cycle?" -- the correct tool is `get_weather` (transport depends on weather), not `schedule_meeting` (the meeting already exists). Tests implicit reasoning.
- **P11:** "Don't check the weather in Antwerp, just find me the quarterly report." -- the correct tool is `search_files`. Calling `get_weather` means the model ignored an explicit negation. Tests instruction following.
- **P12:** "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?" -- the correct tool is `schedule_meeting`. The weather is already provided; calling `get_weather` is redundant. Tests context awareness.

## What we measure

- **Action Score:** correct_tool_calls / 10. How many of the 10 actionable prompts produced valid tool calls with the correct tool. For P10-P12, the tool must match the expected tool. Measures execution capability.
- **Restraint Score:** correct_refusals / 2. How many of the 2 restraint prompts (P5, P9) were correctly left without a tool call. Measures policy calibration.
- **Wrong Tool:** Count of specifically-bad tool calls on P10-P12 (0-3). Each hard prompt has a "wrong tool" that is worse than not calling any tool at all. Measures judgment under misleading context.
- **Reliability:** Average per-prompt (successful_runs / 20). Computed from per-run data *before* majority voting. A model that passes a prompt in 14 of 20 runs gets 0.70 reliability for that prompt, even though majority voting calls it a pass. A stability signal computed from 20 runs per prompt.
- **Multi-Tool Accuracy:** correct_tools / required_tools for P8 (dual-tool prompt). P8 requires both `search_files` and `get_weather`. Ollama's native tool API returns only the first tool call, so this is N/A for native-tools models.
- **Agent Score:** `Action × 0.4 + Restraint × 0.3 + Wrong-Tool-Avoidance × 0.3` where Wrong-Tool-Avoidance = (3 - wrong_tool_count) / 3. A model that calls tools aggressively but picks the wrong ones is penalized. A model that conservatively declines uncertain prompts is rewarded.
- **Latency:** Wall-clock time per inference call (milliseconds).

Everything is run 20 times. Correctness uses majority-vote aggregation; reliability uses per-run data.

> **Context-window caveat:** All Ollama models were run with default settings. Ollama defaults to a 4,096-token context window (`num_ctx`), well below the training context of most models tested (e.g. 131,072 for Qwen 2.5). Our prompts are short enough that 4K is not a binding constraint here, but models may behave differently at longer context lengths or with `num_ctx` tuned to match `n_ctx_train`. Results should be read as "this model at Ollama defaults," not as the model's full capability ceiling.

## Results at a glance

Agent Score rewards correct action **and** correct inaction; wrong-tool calls are penalized. Results below are from Round 3 (20 runs per model, majority-vote aggregation).

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **qwen3:1.7b** | Ollama | native-tools | CN | 0.900 | 1.000 | 0 | **0.960** | 10,665 |
| 2 | **lfm2.5:1.2b** | llama.cpp | openai-compat | US | 0.800 | 1.000 | 0 | **0.920** | 1,567 |
| 3 | qwen3:0.6b | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | **0.880** | 3,410 |
| 4 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.500 | 1.000 | 0 | 0.800 | 2,240 |
| 4 | ministral-3:3b | Ollama | native-tools | FR | 0.500 | 1.000 | 0 | 0.800 | 7,505 |
| 6 | phi4-mini:3.8b | Ollama | raw-schema | US | 0.700 | 1.000 | 1 | 0.780 | 5,180 |
| 7 | gemma3:1b | Ollama | raw-schema | US | 0.600 | 0.500 | 0 | 0.690 | 2,321 |
| 8 | qwen2.5:3b | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.670 | 3,014 |
| 9 | llama3.2:3b | Ollama | native-tools | US | 0.900 | 0.000 | 0 | 0.660 | 1,690 |
| 10 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.640 | 1,015 |
| 10 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.640 | 1,722 |
| 10 | functiongemma | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.640 | 435 |
| 13 | smollm3:3b | Ollama | raw-schema | US | 0.700 | 0.500 | 1 | 0.630 | 9,727 |
| 14 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.600 | 1,549 |
| 14 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0 | 0.600 | 14,157 |
| 16 | jan-v3:4b | Ollama | raw-schema | US | 0.900 | 0.000 | 1 | 0.560 | 2,322 |
| 17 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.700 | 0.500 | 2 | 0.530 | 2,075 |
| 18 | granite3.3:2b | Ollama | native-tools | US | 0.800 | 0.000 | 1 | 0.520 | 1,658 |
| 18 | granite4:3b | Ollama | native-tools | US | 0.800 | 0.000 | 1 | 0.520 | 2,112 |
| 20 | llama3.2:1b | Ollama | native-tools | US | 0.700 | 0.500 | 3 | 0.430 | 1,596 |

### The new #1: qwen3:1.7b

The Round 2 "middle child" turns out to be the benchmark champion. At 3 runs, qwen3:1.7b happened to fail P5 and P9 restraint in 2/3 runs each, producing a misleading Restraint score of 0.500. At 20 runs, it passes both comfortably. It's also the only model to get all three hard prompts right (P10: get_weather, P11: search_files, P12: schedule_meeting) -- Action 0.900 with perfect Restraint and zero wrong tools.

**lfm2.5:1.2b** confirms as the best speed/quality ratio: 0.920 Agent Score at 1,567 ms -- nearly 7x faster than qwen3:1.7b for only 0.040 less. It's the only non-transformer in the top tier.

### Edge agent mini leaderboard (sub-2B models)

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

## What we learned

### Round 1: The original 11 models

The full analysis is in [ROUND1_REPORT.md](ROUND1_REPORT.md). Key findings:

1. **Hard prompts broke the plateau.** In earlier benchmark iterations, four models tied at 0.929. Adding judgment prompts P10-P12 and wrong-tool penalties spread them from 0.570 to 0.800.
2. **Not calling a tool beats calling the wrong one.** qwen2.5:1.5b and ministral-3:3b scored highest by declining uncertain prompts rather than guessing wrong.
3. **Keyword matching is a common failure mode.** Five of eight functional models called `get_weather` whenever they saw "weather" in the prompt, regardless of context -- even when told "don't check the weather."
4. **Bigger isn't always better.** qwen2.5:1.5b outperformed qwen2.5:3b. The relationship between parameter count and agent quality is non-monotonic when judgment is measured.
5. **BitNet 2B-4T produces flawless JSON with ternary weights** and is the only model to handle multi-tool requests, but its tool *selection* judgment on hard prompts is poor.

### Round 2: The community edition

After the Reddit post, 10 community-requested models were added. The full analysis is in [ROUND2_REPORT.md](ROUND2_REPORT.md). Key findings:

1. **Four models tie for #1 at 0.880.** lfm2.5:1.2b, qwen3:0.6b, qwen3:4b, and phi4-mini:3.8b. The fastest is lfm2.5 at 1,470 ms -- a 1.2B state-space hybrid that was initially misranked due to format compliance issues.
2. **Fixing the parser changed the rankings -- in both directions.** Five models needed fallback parsers for non-standard output formats. lfm2.5:1.2b jumped from rank 13 to tied #1 and deepseek-r1:1.5b improved from 0.600 to 0.720 -- both were genuinely capable models hidden behind format issues. But gemma3:1b (0.600 → 0.550) and smollm3:3b (0.740 → 0.710) actually scored *worse* because the parser revealed they were calling tools on restraint prompts. Format-blind benchmarks can both underestimate and overestimate models.
3. **Parameter count is a weak predictor.** Qwen3 family rankings are non-monotonic: 0.6B (0.880) > 4B (0.880) > 1.7B (0.670). A 1.2B state-space model matches 3.8B transformers. Architecture and training data matter more than raw size.
4. **Purpose-built doesn't mean best.** functiongemma (270M, fine-tuned for function calling) is the fastest model (476 ms) with perfect restraint, but falls into the same keyword traps as generic models on hard prompts.
5. **Generational improvement is real.** granite4:3b (0.670) vs granite3.3:2b (0.480) shows clear improvement within IBM's model line. SmolLM3 matches SmolLM2 with better multi-tool support.
6. **3-run majority voting has high variance on hard prompts.** bitnet-2B-4T shifted from 0.570 to 0.810 between Round 1 and Round 2 reruns, entirely due to different outcomes on P10 and P12.
7. **Thinking mode is a double-edged sword.** qwen3:4b spends 63 seconds average per prompt thinking, for the same score as the 0.6B at 3.6 seconds. For tool-calling decisions, longer thinking chains don't consistently help.

### Round 3: The 20-run validation

Round 3 reran all 20 models with 20 runs each to eliminate small-sample variance. The full analysis is in [ROUND3_REPORT.md](ROUND3_REPORT.md). Key findings:

1. **qwen3:1.7b is the benchmark champion at 0.960.** Jumped from 11th to 1st. Its Round 2 restraint failures were borderline calls that happened to go wrong in 2/3 runs. At 20 runs, it passes P5 and P9 comfortably and is the only model to get all three hard prompts right.
2. **The Round 2 four-way tie was a sampling artifact.** Those four models now score 0.960, 0.920, 0.880, and 0.780 -- spread across a 0.180 range.
3. **Seven models changed by more than 0.05.** bitnet-2B-4T dropped 0.280, phi4-mini dropped 0.100, granite4 dropped 0.150. In the other direction, qwen3:1.7b gained 0.290 and gemma3:1b gained 0.140.
4. **deepseek-r1:1.5b effectively doesn't work for tool calling.** Action 0.000 at 20 runs -- it never reliably produces a parseable tool call. Its Round 2 score was a fluke.
5. **The gap between "works" and "works reliably" is wider than expected.** Many models hover around 50% call rate on borderline prompts. At 3 runs, a coin flip decides their score. At 20 runs, the truth emerges.

## The bottom line

After testing 20 models across three rounds -- 4,800 total inference calls on CPU -- the picture is clearer and more nuanced than early results suggested.

**Local tool-calling agents work today on commodity hardware**, and they're better than expected. Two models exceed 0.900 Agent Score, with lfm2.5:1.2b doing it in 1.6 seconds. Simple, unambiguous tool dispatch is a solved problem at every size from 270M up.

**One model can do it all.** qwen3:1.7b (0.960) is the only model to combine high action (0.900), perfect restraint, and zero wrong tools -- including all three hard prompts. This wasn't visible at 3 runs. The benchmark's hardest prompt (P12) was previously unsolvable; qwen3:1.7b solves it reliably.

**Sample size matters more than most benchmarks acknowledge.** The Round 2 leaderboard was substantially wrong on 7 of 21 models. Rankings based on 3 runs should be treated as preliminary. For tool-calling benchmarks at temperature > 0, 10-20 runs is the minimum for stable results on borderline prompts.

**How you parse matters as much as what you test.** Five models needed fallback parsers for non-standard output formats. Format-blind benchmarks can both underestimate models (lfm2.5: 0.640 → 0.920) and overestimate them (gemma3: 0.600 → 0.690 after a complex journey through parser fixes).

For anyone building a local agent pipeline:

- **For judgment-sensitive tasks:** qwen3:1.7b (0.960, 10.7s) is the top recommendation if you can tolerate the latency. lfm2.5:1.2b (0.920, 1.6s) is the best choice if speed matters -- nearly 7x faster for 0.040 less score. It requires a bracket-notation parser. qwen3:0.6b (0.880, 3.4s) is the best Ollama-native option.
- **For routing clear-cut requests:** Almost any functional model works. functiongemma does it at 435ms. The problem is solved.
- **For latency-critical deployments under 1 second:** functiongemma (435ms, 0.640) and qwen2.5:0.5b (1,015ms, 0.640) are the only options. Both fail on judgment traps.
- **Full autonomy is still premature at this model size.** Even the best model misses 10% of actionable prompts. The failure mode to guard against isn't "model refuses to act" -- it's "model confidently takes the wrong action." Confirmation prompts for destructive actions remain necessary.
- **Test your actual prompts.** Rankings here are specific to this prompt set, this scoring formula, and these model-protocol pairs. Run your own prompts before trusting any leaderboard, including this one.

## Caveats and limitations

This benchmark has a narrow scope by design, and the results should be interpreted accordingly:

- **Small prompt set.** 12 prompts (3 of which test judgment) is enough to reveal patterns but not enough to make strong statistical claims. Confirming the failure modes observed would require a larger and more varied prompt set.
- **Safety-weighted scoring.** The Agent Score gives 60% combined weight to restraint and wrong-tool-avoidance, structurally favoring conservative models. Under an action-maximizing formula, aggressive models like llama3.2:3b (Action 0.900) and bitnet-2B-4T (Action 0.900) would rank much higher. The scoring reflects one deployment preference, not a universal truth.
- **Model-protocol pairs, not models in isolation.** Each result reflects a specific model running through a specific backend (Ollama native tools, Ollama raw prompt, llama.cpp, or BitNet). The same model may behave very differently with a different interaction contract -- phi4-mini's score jumped dramatically when switched from native tools to raw prompt in Round 1. Rankings should not be read as generalizing across protocols.
- **Twenty runs per prompt (Round 3).** Majority voting now stabilizes most prompts. The upgrade from 3 to 20 runs changed 7 of 20 models by more than 0.05, confirming that 3 runs was insufficient. Twenty runs is adequate for this prompt set but models near 50% call rate on specific prompts may still fluctuate slightly.
- **Format compliance affects scores -- and we fixed five cases.** Five models needed fallback parsers across two rounds: lfm2.5 (bracket notation), jan-v3 (bare JSON), gemma3 (funcall-in-tags), deepseek-r1 (bare funcalls), and smollm3 (mixed formats). The fixes improved some scores (lfm2.5: 0.640 → 0.880, deepseek-r1: 0.600 → 0.720) but revealed hidden problems in others (gemma3: 0.600 → 0.550, smollm3: 0.740 → 0.710). Format-blind parsing can flatter a model by hiding restraint failures. Scores partly reflect format training, and benchmarks should consider model-specific parsers.
- **Default Ollama settings.** All Ollama models ran with default `num_ctx` (4,096 tokens) and default sampling parameters (temperature, top_p, etc.). Our prompts are short enough that context isn't a binding constraint, but results reflect "model at Ollama defaults," not full capability.
- **CPU-only, single machine.** All inference ran on one AMD Ryzen AI 7 350. Latency numbers are specific to this hardware and would differ on other CPUs or with GPU acceleration. Relative rankings should be more stable than absolute latencies.
- **No multi-turn evaluation.** All prompts are single-turn. Real agent pipelines involve multi-turn conversations where the model receives tool results and decides what to do next. Single-turn tool dispatch is a necessary but not sufficient condition for agent viability.

## Run it yourself

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- Python 3.10+
- ~20 GB free disk space for models
- For BitNet: `cmake`, `clang`, `clang++`, and ~14 GB additional disk space during build

### Quick start (Ollama models only)

If you just want to test the 9 Ollama models and skip BitNet:

```bash
# Pull the models
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama pull smollm2:1.7b
ollama pull ministral-3:3b
ollama pull deepseek-r1:1.5b
ollama pull gemma3:1b
ollama pull phi4-mini:3.8b

# Clone and set up
git clone <this-repo> && cd local-agent-bench
python -m venv .venv
source .venv/bin/activate
pip install ollama requests

# Comment out the bitnet entries in ALL_MODELS and the
# start/stop_bitnet_server() calls in main(), then run:
python bench.py
```

### Full setup (including BitNet)

```bash
# 1. Pull Ollama models (same as above)

# 2. Clone and build BitNet
cd ~/projects
git clone https://github.com/microsoft/BitNet.git bitnet
cd bitnet
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate

# Relax the torch version constraint for Python 3.12+
sed -i 's/torch~=2.2.1/torch>=2.2.1/g' \
  3rdparty/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  3rdparty/llama.cpp/requirements/requirements-convert_hf_to_gguf_update.txt

pip install -r requirements.txt
pip install --no-deps 3rdparty/llama.cpp/gguf-py

# Download and build the base 3B model
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-3B -q i2_s

# Download the instruction-tuned 2B-4T model
pip install huggingface_hub
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T

# If compile fails with a const-correctness error in ggml-bitnet-mad.cpp,
# change line 811 from "int8_t * y_col" to "const int8_t * y_col" and rebuild:
#   cmake --build build --config Release

# Verify
ls build/bin/llama-server
ls models/bitnet_b1_58-3B/ggml-model-i2_s.gguf
ls models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# 3. Clone and run the benchmark
cd ~/projects
git clone <this-repo> && cd local-agent-bench
python -m venv .venv
source .venv/bin/activate
pip install ollama requests

# Update BITNET_DIR in lib/bitnet_backend.py if your BitNet path differs
python bench.py
```

The full run (20 models x 12 prompts x 20 runs = 4,800 inference calls) takes roughly 5 hours on the hardware described above. Most of that time is qwen3:1.7b (thinking mode), BitNet 3B (base model), smollm3:3b, and ministral-3:3b -- the other models finish faster. For a quick test, use `--num-runs 3` (runs in ~45 minutes).

### Customising

The entry point is `bench.py`; supporting modules live in `lib/`. To add models, prompts, or adjust runs:

- **Add an Ollama model (native tool API):** Add `{"name": "your-model:tag", "backend": "ollama", "origin": "XX"}` to `ALL_MODELS` in `lib/bench_config.py`
- **Add an Ollama model (raw prompt):** Use `"backend": "ollama_raw"` for models that don't support Ollama's native tool API or perform better with system-prompt-based tool calling
- **Add a prompt:** Append to `TEST_PROMPTS` in `lib/bench_config.py`. If it's a restraint prompt (correct answer is no tool call), add its 0-based index to `RESTRAINT_INDICES`. If it's a hard prompt with an expected tool, add it to `EXPECTED_TOOLS` and `WRONG_TOOL_MAP`
- **Change run count:** Edit `num_runs` in `main()` in `bench.py`
- **Add tools:** Extend the `TOOLS` list in `lib/bench_config.py` and `TOOL_DISPATCH` dict in `bench.py`. Update `BITNET_SYSTEM_PROMPT` in `lib/bitnet_backend.py` if you want raw-prompt models to know about them
- **Add to edge leaderboard:** Add the model name to `EDGE_MODELS` in `lib/bench_config.py`

## License

Use it freely; attribution appreciated. It's a benchmark script, not a product.
