# Local Agent Bench

**Can a $1,000 laptop run an AI agent that knows when to use tools -- and when not to?**

I tested 21 small open-weight models locally on CPU to see which ones can act -- and which ones know when not to. No cloud API. No GPU. Just Ollama, a handful of 1-bit and 4-bit quantised models, and a Framework 13 running Arch Linux.

[Round 1](ROUND1_REPORT.md) tested 11 models from 7 organisations. After the post [went viral on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1qyg10z/) , [Round 2](ROUND2_REPORT.md) added 10 community-requested models -- including every model that was suggested in the comments.

The motivation is practical. Local and private AI agents are increasingly attractive -- no per-token costs, no data leaving the machine, no vendor lock-in. But an agent that acts incorrectly is worse than one that does nothing: a wrong API call costs money, sends the wrong message, or deletes the wrong file. The hard problem isn't generating well-formed JSON. It's deciding whether to act at all.

This benchmark measures **judgment** -- whether a model knows *when* to call a tool -- not just **execution** -- whether it can format a tool call correctly.

## TL;DR

- **Four models tie for #1** at 0.880 Agent Score: lfm2.5:1.2b (a 1.2B state-space hybrid), qwen3:0.6b, qwen3:4b, and phi4-mini:3.8b. The fastest is lfm2.5 at 1,470 ms.
- **lfm2.5:1.2b jumped from rank 13 to #1** after adding a parser for its bracket-notation output format. The model was always making correct decisions; the benchmark couldn't see them. A 50-line parser fix revealed a top-tier model.
- Every model that successfully emitted tool calls can handle simple, unambiguous tool calls on CPU at 1-8s latency.
- When prompts require judgment -- resisting keyword triggers, respecting negation, noticing redundant information -- most sub-4B models fail.
- The top-scoring models win by *declining to act* when uncertain, not by calling more tools. Under safety-weighted scoring, conservatism beats aggression.
- Parameter count is a weak predictor of tool-calling quality. Rankings within the Qwen3 family are non-monotonic: 0.6B > 4B > 1.7B. A 1.2B state-space model ties 3.8B transformers.
- A purpose-built function-calling model (functiongemma, 270M) is the fastest in the benchmark but falls into the same keyword traps as generic models 7x its size.
- P12 ("The weather is 8°C and rainy. Should I schedule a meeting?") is the hardest prompt: only 3 of 21 models call the correct tool.
- **Format compliance masks true behavior -- in both directions.** Parser fixes for 5 models revealed that format-blind scoring both underestimates (lfm2.5: 0.640 → 0.880) and overestimates (gemma3: 0.600 → 0.550) models by hiding their actual tool-calling behavior.
- 3-run majority voting has high variance on hard prompts: bitnet-2B-4T shifted from 0.570 to 0.810 between runs due to different P10/P12 outcomes.

## Why this exists

Tool-calling is the backbone of AI agents. An LLM that can reliably decide "this prompt needs `get_weather`, that one needs `schedule_meeting`, and this other one needs *nothing at all*" is the difference between a useful agent and an expensive autocomplete.

But there's a harder question: when a prompt mentions "weather" but the correct action is *not* to call `get_weather`, can the model resist the keyword trigger? When the user says "don't check the weather, just find the report," does the model listen? When the weather is already provided in the prompt, does the model notice?

Cloud models handle this well. But what about local models running on your laptop's CPU? The small open-weight models (0.5B-3.8B parameters) that Ollama makes trivially easy to run -- can they actually *do* this?

This benchmark tests all of that: 21 models from 10 organisations across 4 countries, 12 prompts, 3 runs each, on a machine with no discrete GPU.

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

**Qwen 3 (4B, 1.7B, 0.6B) -- the most-requested family.** Six separate users asked for Qwen3. All three sizes have built-in thinking capability. The 0.6B is the smallest model in the benchmark with native tool support. Tests whether the Qwen2.5 → Qwen3 generation jump matters for tool calling.

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
- **Reliability:** Average per-prompt (successful_runs / 3). Computed from per-run data *before* majority voting. A model that passes a prompt in 2 of 3 runs gets 0.67 reliability for that prompt, even though majority voting calls it a pass. A coarse stability signal from a 3-run sample, not a deployment-grade confidence estimate.
- **Multi-Tool Accuracy:** correct_tools / required_tools for P8 (dual-tool prompt). P8 requires both `search_files` and `get_weather`. Ollama's native tool API returns only the first tool call, so this is N/A for native-tools models.
- **Agent Score:** `Action × 0.4 + Restraint × 0.3 + Wrong-Tool-Avoidance × 0.3` where Wrong-Tool-Avoidance = (3 - wrong_tool_count) / 3. A model that calls tools aggressively but picks the wrong ones is penalized. A model that conservatively declines uncertain prompts is rewarded.
- **Latency:** Wall-clock time per inference call (milliseconds).

Everything is run 3 times. Correctness uses majority-vote aggregation; reliability uses per-run data.

> **Context-window caveat:** All Ollama models were run with default settings. Ollama defaults to a 4,096-token context window (`num_ctx`), well below the training context of most models tested (e.g. 131,072 for Qwen 2.5). Our prompts are short enough that 4K is not a binding constraint here, but models may behave differently at longer context lengths or with `num_ctx` tuned to match `n_ctx_train`. Results should be read as "this model at Ollama defaults," not as the model's full capability ceiling.

## Results at a glance

Agent Score rewards correct action **and** correct inaction; wrong-tool calls are penalized. Results below are from the latest run (Round 2, all 21 models rerun fresh).

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **lfm2.5:1.2b** | llama.cpp | openai-compat | US | 0.700 | 1.000 | 0 | **0.880** | 1,470 |
| 1 | phi4-mini:3.8b | Ollama | raw-schema | US | 0.700 | 1.000 | 0 | **0.880** | 5,460 |
| 1 | **qwen3:0.6b** | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | **0.880** | 3,645 |
| 1 | **qwen3:4b** | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | **0.880** | 63,717 |
| 5 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 0 | **0.840** | 2,211 |
| 6 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.900 | 0.500 | 0 | 0.810 | 2,036 |
| 7 | ministral-3:3b | Ollama | native-tools | FR | 0.500 | 1.000 | 0 | 0.800 | 7,157 |
| 8 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 1 | 0.740 | 1,626 |
| 9 | **deepseek-r1:1.5b** | Ollama | raw-schema | CN | 0.300 | 1.000 | 0 | 0.720 | 1,672 |
| 10 | **smollm3:3b** | Ollama | raw-schema | US | 0.900 | 0.500 | 1 | 0.710 | 12,096 |
| 11 | qwen2.5:3b | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.670 | 2,801 |
| 11 | **qwen3:1.7b** | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.670 | 11,903 |
| 11 | **granite4:3b** | Ollama | native-tools | US | 0.800 | 0.500 | 1 | 0.670 | 2,402 |
| 14 | llama3.2:3b | Ollama | native-tools | US | 0.900 | 0.000 | 0 | 0.660 | 1,726 |
| 15 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.640 | 881 |
| 15 | **functiongemma** | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.640 | 476 |
| 17 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0 | 0.600 | 11,362 |
| 18 | **jan-v3:4b** | Ollama | raw-schema | US | 0.900 | 0.000 | 1 | 0.560 | 2,335 |
| 19 | **gemma3:1b** | Ollama | raw-schema | US | 0.500 | 0.500 | 1 | 0.550 | 2,426 |
| 20 | **granite3.3:2b** | Ollama | native-tools | US | 0.700 | 0.000 | 1 | 0.480 | 1,650 |
| 21 | **llama3.2:1b** | Ollama | native-tools | US | 0.700 | 0.500 | 3 | 0.430 | 1,461 |

**Bold** = new in Round 2.

### The surprising result: a 1.2B state-space model ties for #1

Four models share the top score (0.880): lfm2.5:1.2b, phi4-mini:3.8b, qwen3:0.6b, and qwen3:4b. The biggest surprise is **lfm2.5:1.2b** -- Liquid AI's state-space hybrid that initially scored 0.640 because the parser couldn't read its bracket-notation output (`[get_weather(city="Antwerp")]`). After adding a fallback parser, it jumped to 0.880: perfect restraint, zero wrong tools, and the fastest latency at the top tier (1,470 ms). It's the only non-transformer architecture in the #1 tier.

qwen3:0.6b -- at just 600 million parameters -- also ties for #1. The Qwen3 family rankings are non-monotonic: 0.6B > 4B > 1.7B. The 1.7B sits in a capability valley -- large enough to be aggressive about calling tools, not large enough to exercise judgment about when not to.

### Edge agent mini leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Action | Restraint | Wrong Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|
| 1 | **lfm2.5:1.2b** | llama.cpp | openai-compat | 0.700 | 1.000 | 0 | **0.880** | 1,470 |
| 1 | **qwen3:0.6b** | Ollama | native-tools | 0.700 | 1.000 | 0 | **0.880** | 3,645 |
| 3 | qwen2.5:1.5b | Ollama | native-tools | 0.600 | 1.000 | 0 | **0.840** | 2,211 |
| 4 | bitnet-2B-4T | bitnet.cpp | openai-compat | 0.900 | 0.500 | 0 | 0.810 | 2,036 |
| 5 | smollm2:1.7b | Ollama | native-tools | 0.600 | 1.000 | 1 | 0.740 | 1,626 |
| 6 | **deepseek-r1:1.5b** | Ollama | raw-schema | 0.300 | 1.000 | 0 | 0.720 | 1,672 |
| 7 | **qwen3:1.7b** | Ollama | native-tools | 0.800 | 0.500 | 1 | 0.670 | 11,903 |
| 8 | qwen2.5:0.5b | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.640 | 881 |
| 8 | **functiongemma** | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.640 | 476 |
| 10 | **gemma3:1b** | Ollama | raw-schema | 0.500 | 0.500 | 1 | 0.550 | 2,426 |
| 11 | **llama3.2:1b** | Ollama | native-tools | 0.700 | 0.500 | 3 | 0.430 | 1,461 |

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

## The bottom line

After testing 21 models across two rounds -- 756 total inference calls on CPU -- the picture is clearer than it was with 11.

**Local tool-calling agents work today on commodity hardware**, and they're better than expected. Four models achieve 0.880 Agent Score, with the fastest (lfm2.5:1.2b) doing it in 1.5 seconds. Simple, unambiguous tool dispatch is a solved problem at every size from 270M up.

**The judgment gap is real but narrowing.** In Round 1, no model reliably handled all three judgment dimensions (keyword resistance, negation following, context awareness). In Round 2, four models handle two of three with perfect restraint and zero wrong tools. P12 (context awareness) remains the hardest: only 3 of 21 models get it right. But the trajectory from qwen2.5 to qwen3 suggests the next generation may close this gap.

**The conservative strategy keeps winning**, and the community-requested models confirmed rather than overturned this finding. The top models in both rounds share the same pattern: perfect restraint, zero wrong tools, moderate action. Under a scoring formula that penalizes wrong actions more than missed ones, knowing when *not* to act is the dominant skill. This isn't a universal truth -- it reflects a specific deployment preference -- but it's the right default for autonomous agents where wrong actions have consequences.

**How you parse matters as much as what you test.** Five models needed fallback parsers for non-standard output formats. lfm2.5:1.2b jumped from rank 13 to tied #1 after a bracket-notation parser was added. But the lesson isn't just about underestimation: gemma3:1b and smollm3:3b actually scored *worse* after parser fixes, because the old parser was hiding their restraint failures. Format-blind benchmarks can flatter models by missing bad behavior just as easily as they penalize good behavior. Any tool-calling benchmark should consider model-specific output formats or risk conflating format training with reasoning capability.

For anyone building a local agent pipeline:

- **For routing clear-cut requests:** Almost any functional model works. qwen2.5:0.5b handles "check the weather in Paris" at sub-1s latency. functiongemma does it at 476ms. The problem is solved.
- **For judgment-sensitive tasks:** lfm2.5:1.2b is the new top recommendation. 1.2B parameters, 1.5s average latency, 0.880 Agent Score with perfect restraint and multi-tool support. It requires a bracket-notation parser (not standard `<tool_call>` tags). If you need standard output format, qwen3:0.6b (0.880, 3.6s) is the best Ollama-native choice. qwen2.5:1.5b (0.840, 2.2s) is a strong alternative if you want faster inference without a custom parser.
- **For latency-critical deployments under 1 second:** functiongemma (476ms, 0.640) and qwen2.5:0.5b (881ms, 0.640) are the only options. Both handle easy prompts well but fail on judgment traps.
- **For maximum action rate:** bitnet-2B-4T (Action 0.900, Multi-Tool 1.000) and llama3.2:3b (Action 0.900) call tools most aggressively. Both have restraint problems -- llama3.2 calls a tool on every prompt without exception. Use these only with a human-in-the-loop or a strict safety layer.
- **Full autonomy is still premature at this model size.** Even the best model (qwen3:0.6b) misses 30% of actionable prompts. The failure mode to guard against isn't "model refuses to act" -- it's "model confidently takes the wrong action." Confirmation prompts for destructive actions, tool allowlists, or escalation to a larger model for ambiguous prompts remain necessary.
- **Test your actual prompts.** Rankings here are specific to this prompt set, this scoring formula, and these model-protocol pairs. Run your own prompts before trusting any leaderboard, including this one.

## Caveats and limitations

This benchmark has a narrow scope by design, and the results should be interpreted accordingly:

- **Small prompt set.** 12 prompts (3 of which test judgment) is enough to reveal patterns but not enough to make strong statistical claims. Confirming the failure modes observed would require a larger and more varied prompt set.
- **Safety-weighted scoring.** The Agent Score gives 60% combined weight to restraint and wrong-tool-avoidance, structurally favoring conservative models. Under an action-maximizing formula, aggressive models like llama3.2:3b (Action 0.900) and bitnet-2B-4T (Action 0.900) would rank much higher. The scoring reflects one deployment preference, not a universal truth.
- **Model-protocol pairs, not models in isolation.** Each result reflects a specific model running through a specific backend (Ollama native tools, Ollama raw prompt, llama.cpp, or BitNet). The same model may behave very differently with a different interaction contract -- phi4-mini's score jumped dramatically when switched from native tools to raw prompt in Round 1. Rankings should not be read as generalizing across protocols.
- **Three runs per prompt.** Majority voting stabilizes easy prompts but not hard ones. bitnet-2B-4T's Agent Score shifted by 0.240 between Round 1 and Round 2 reruns, entirely from different P10/P12 outcomes. Models near decision boundaries on hard prompts will fluctuate.
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

The full run (21 models x 12 prompts x 3 runs = 756 inference calls) takes roughly 2-3 hours on the hardware described above. Most of that time is qwen3:4b (thinking mode), BitNet 3B (base model), and ministral-3:3b -- the other models finish faster.

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
