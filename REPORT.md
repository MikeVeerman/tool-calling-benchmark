# Local LLM Tool-Calling Benchmark Report

**Date:** 2026-02-06
**Runs:** 3 per model/prompt combination (162 total inference calls)
**Hardware:** CPU-only (no GPU acceleration)

## Models Tested

| Model | Backend | Size | Notes |
|---|---|---|---|
| qwen2.5:3b | Ollama | 3B params | Instruction-tuned, native tool-calling |
| qwen2.5:1.5b | Ollama | 1.5B params | Instruction-tuned, native tool-calling |
| qwen2.5:0.5b | Ollama | 0.5B params | Instruction-tuned, native tool-calling |
| llama3.2:3b | Ollama | 3B params | Instruction-tuned, native tool-calling |
| smollm2:1.7b | Ollama | 1.7B params | Instruction-tuned, native tool-calling |
| bitnet-3B | BitNet (llama-server) | 3B params | 1.58-bit quantised base model, text-prompted tool schema |

## Tools Available

Three tools were defined and provided to all models:

1. **get_weather**(city) -- returns mock weather data for a city
2. **search_files**(pattern) -- returns mock file search results for a glob pattern
3. **schedule_meeting**(title, time, attendees?) -- returns mock meeting confirmation

Ollama models received tools via the native Ollama tool-calling API. BitNet received a system prompt embedding the tool schemas in text, instructed to respond with `<tool_call>` tags.

## Test Prompts

| ID | Prompt | Category | Expected Behavior |
|---|---|---|---|
| P1 | "What's the weather in Antwerp?" | Obvious single-tool | Call get_weather("Antwerp") |
| P2 | "Find all Python files in the project." | Obvious different tool | Call search_files("*.py") |
| P3 | "Schedule a meeting called 'Sprint Review' for 2025-02-10T14:00:00 with alice@co.com and bob@co.com." | Multi-argument extraction | Call schedule_meeting with title, time, attendees |
| P4 | "I'm heading to Brussels tomorrow, anything I should know?" | Ambiguous | Acceptable to call get_weather or not |
| P5 | "What tools do you have access to?" | **RESTRAINT** | Should NOT call a tool |
| P6 | "What's the weather in the city where we have our next sprint review?" | Multi-step reasoning | Reasonable to call get_weather with a guess, or decline (no memory context) |
| P7 | "Oh hey, could you maybe like set up a meeting -- 'Q3 Roadmap' -- for next Tuesday at 3pm? I think dave@co.com and maybe susan@co.com should come" | Noisy parameter extraction | Call schedule_meeting with extracted params from messy language |
| P8 | "Search for all files matching '*.py' and also tell me the weather in Paris." | Adversarial dual-tool | Tests if model picks one tool or handles the dual request |
| P9 | "Can you write a Python script that checks the weather using an API?" | **RESTRAINT** | Should NOT call get_weather -- asking for code, not a tool call |

## Averaged Summary (3 runs, majority-vote)

| Model | Tool Calls | Valid Args | Avg Latency | Restraint |
|---|---|---|---|---|
| qwen2.5:3b | 7/9 | 7/7 | 3,927 ms | 1/2 |
| qwen2.5:1.5b | 5/9 | 5/5 | 2,886 ms | 2/2 |
| qwen2.5:0.5b | 6/9 | 6/6 | 1,487 ms | 2/2 |
| llama3.2:3b | 9/9 | 9/9 | 2,428 ms | 0/2 |
| smollm2:1.7b | 6/9 | 6/6 | 2,549 ms | 2/2 |
| bitnet-3B | 0/9 | 0/0 | 19,162 ms | 2/2 |

## Per-Prompt Averaged Latency (ms)

| Model | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b | 9,642 | 1,324 | 4,035 | 3,141 | 3,776 | 1,762 | 4,417 | 2,560 | 4,690 |
| qwen2.5:1.5b | 5,405 | 769 | 2,410 | 4,161 | 2,918 | 1,589 | 2,528 | 1,397 | 4,800 |
| qwen2.5:0.5b | 1,067 | 410 | 1,042 | 1,968 | 1,559 | 618 | 1,152 | 653 | 4,913 |
| llama3.2:3b | 4,047 | 1,233 | 3,541 | 1,237 | 2,681 | 1,533 | 4,039 | 2,195 | 1,345 |
| smollm2:1.7b | 7,081 | 1,472 | 3,735 | 670 | 598 | 1,606 | 4,053 | 3,048 | 676 |
| bitnet-3B | 19,433 | 20,176 | 20,678 | 16,354 | 20,043 | 20,288 | 21,136 | 14,793 | 19,555 |

## Per-Prompt Averaged Tool-Call Behavior

| Model | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b | get_weather | search_files | schedule_meeting | get_weather | -- | -- | schedule_meeting | search_files | get_weather |
| qwen2.5:1.5b | get_weather | search_files | schedule_meeting | -- | -- | -- | schedule_meeting | search_files | -- |
| qwen2.5:0.5b | get_weather | search_files | schedule_meeting | -- | -- | get_weather | schedule_meeting | search_files | -- |
| llama3.2:3b | get_weather | search_files | schedule_meeting | get_weather | search_files | get_weather | schedule_meeting | search_files | search_files |
| smollm2:1.7b | get_weather | search_files | schedule_meeting | -- | -- | get_weather | schedule_meeting | search_files | -- |
| bitnet-3B | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Detailed Analysis

### Tier 1: Best overall -- qwen2.5:1.5b and smollm2:1.7b

Both models achieved the best balance of accuracy and restraint:

- **6/9 or 5/9 tool calls** with **100% valid arguments** when tools were invoked
- **2/2 restraint** -- correctly declined to call tools on both P5 (meta question) and P9 (code-writing request)
- Appropriately conservative on P4 (ambiguous) and P6 (under-specified context)

**qwen2.5:1.5b** was slightly more conservative (5/9 calls, skipped P6) with moderate latency (~2.9s avg). **smollm2:1.7b** was a surprise performer at 6/9 calls with perfect restraint and the fastest "no tool" responses (~600ms when declining).

### Tier 2: Aggressive but accurate -- llama3.2:3b and qwen2.5:3b

**llama3.2:3b** achieved the highest raw tool-call rate (9/9) with 100% valid arguments, but this came at the cost of **0/2 restraint**. It called tools on every single prompt, including the meta question (P5) and the code-writing request (P9). On P5 it called `search_files`, and on P9 it called `search_files` -- neither makes sense. This model is a "tool-call hammer" that treats every prompt as a nail.

**qwen2.5:3b** showed the same pattern to a lesser degree (7/9 calls, 1/2 restraint). It correctly restrained on P5 but failed P9 by calling `get_weather` -- triggered by the word "weather" in the code-writing prompt. Run-to-run consistency was mixed: it passed P9 restraint in Run 1 but failed in Runs 2 and 3.

### Tier 3: Fast but limited -- qwen2.5:0.5b

The smallest Ollama model was the **fastest** by far (1,487ms average) and showed **perfect restraint** (2/2). However, its responses were more variable run-to-run:

- P1 latency swung from 2,404ms (Run 1) down to 393ms (Runs 2-3), likely due to model warm-up
- P6 behavior was inconsistent: called get_weather in Runs 1-2 but not Run 3
- P9 latency was disproportionately high (3.6-7.4s) suggesting it struggled with the code-writing prompt before ultimately (correctly) declining

For simple, single-tool prompts it was excellent. The 0.5B size shows its limits on harder prompts.

### Tier 4: Non-functional -- bitnet-3B

BitNet produced **zero successful tool calls** across all 27 attempts. Its output was incoherent text fragments. Example raw output for P1 ("What's the weather in Antwerp?"):

```
4 back an half. (
Bar. a. with. a. have. E. ever (
The. lots - ex, ( (
 ( all
 ex dec - to) ( ex ex the )
```

This was expected. The bitnet_b1_58-3B model is a **base (non-instruction-tuned) 1.58-bit quantised model**. It was not trained to follow instructions, understand tool schemas, or produce structured output. The system prompt embedding tool definitions had no effect.

Latency was consistently ~20s per prompt (hitting the 512 max_tokens generation limit), roughly 8x slower than the Ollama models running on the same CPU hardware.

### Per-Run Consistency

Most Ollama models were highly consistent across runs. Notable exceptions:

| Model | Prompt | Variation |
|---|---|---|
| qwen2.5:3b | P9 | Restrained in R1, called get_weather in R2-R3 |
| qwen2.5:1.5b | P6 | Called get_weather in R1, declined in R2-R3 |
| qwen2.5:0.5b | P6 | Called get_weather in R1-R2, declined in R3 |
| llama3.2:3b | P4 | Called get_weather in R1-R2, declined in R3 |
| llama3.2:3b | P5 | search_files in R1/R3, get_weather in R2 |
| smollm2:1.7b | P8 | search_files in R1-R2, schedule_meeting in R3 |

The ambiguous prompts (P4, P6) and the P9 restraint test produced the most run-to-run variation, which is expected -- these prompts sit on decision boundaries.

### The Harder Prompts (P6-P9)

**P6 (multi-step reasoning):** Most models either called get_weather with an implied city or correctly declined due to insufficient context. No model attempted to chain reasoning (e.g., there's no "lookup meeting" tool anyway). qwen2.5:0.5b and smollm2 called get_weather, while qwen2.5:3b and qwen2.5:1.5b mostly declined.

**P7 (noisy parameter extraction):** Every Ollama model successfully extracted the meeting parameters from the messy natural language. All called schedule_meeting with the correct tool. This was a strong showing across the board -- even the 0.5B model handled it.

**P8 (adversarial dual-tool):** All Ollama models picked exactly one tool (search_files in most cases, schedule_meeting for smollm2 in one run). None attempted to call two tools or hallucinate a dual response. The Ollama API only returns the first tool call, so this tests which tool the model prioritises -- file search consistently won.

**P9 (tool-adjacent trick):** The word "weather" in a code-writing request was a strong trigger. llama3.2:3b always called a tool (search_files). qwen2.5:3b failed in 2/3 runs (calling get_weather). The smaller models (qwen2.5:1.5b, qwen2.5:0.5b, smollm2:1.7b) all correctly restrained.

## Conclusions

1. **Model size does not predict restraint quality.** The 0.5B and 1.5B models showed better restraint than the 3B models. Smaller models may be more conservative by default, which happens to be the correct behavior for ambiguous or trick prompts.

2. **llama3.2:3b is a tool-call maximalist.** It will call a tool on virtually any prompt. This is useful if you want aggressive tool use, but dangerous in production where false-positive tool calls waste resources or cause side effects.

3. **smollm2:1.7b is a strong contender.** Despite being a newer/smaller model, it matched the Qwen models on accuracy and beat them on restraint, with competitive latency.

4. **BitNet is not viable for tool-calling.** The 1.58-bit base model cannot follow instructions or produce structured output. An instruction-tuned BitNet variant would be needed for a fair comparison.

5. **Noisy NLP extraction (P7) is a solved problem** at this scale. Every Ollama model handled it without issue.

6. **The "weather in code" trick (P9) is the best discriminator.** It cleanly separates models that understand intent from those that pattern-match on keywords.
