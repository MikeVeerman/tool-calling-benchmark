# Local LLM Tool-Calling Benchmark Report (Round 3)

**Date:** 2026-02-06
**Runs:** 3 per model/prompt combination (297 total inference calls)
**Hardware:** CPU-only (no GPU acceleration)

## Machine Specs

| Component | Detail |
|---|---|
| CPU | AMD Ryzen AI 7 350 w/ Radeon 860M |
| Cores / Threads | 8 cores / 16 threads |
| Architecture | x86_64 (Zen 5, Strix Point) |
| CPU Max Clock | 2.0 GHz (boost-enabled) |
| RAM | 32 GB DDR5 (30 Gi usable) |
| GPU | Integrated Radeon 860M (not used for inference) |
| OS | Arch Linux, kernel 6.18.3-arch1-1 |
| ISA Extensions | AVX-512, AVX2, SSE4.2 |

All inference ran on CPU only. Ollama models use llama.cpp under the hood with Q4_K_M quantization by default. BitNet models use Microsoft's bitnet.cpp with native 1.58-bit (I2_S) kernels.

## Models Tested

| Model | Params | Backend | Origin | Notes |
|---|---|---|---|---|
| qwen2.5:3b | 3B | Ollama (native tools) | CN (Alibaba) | Instruction-tuned, Ollama native tool-calling API |
| qwen2.5:1.5b | 1.5B | Ollama (native tools) | CN (Alibaba) | Instruction-tuned, Ollama native tool-calling API |
| qwen2.5:0.5b | 0.5B | Ollama (native tools) | CN (Alibaba) | Smallest Qwen, instruction-tuned |
| llama3.2:3b | 3B | Ollama (native tools) | US (Meta) | Instruction-tuned, Ollama native tool-calling API |
| smollm2:1.7b | 1.7B | Ollama (native tools) | US (HuggingFace) | Instruction-tuned, Ollama native tool-calling API |
| ministral-3:3b | 3B | Ollama (native tools) | FR (Mistral) | Mistral's edge model, Apache 2.0 |
| deepseek-r1:1.5b | 1.5B | Ollama (raw prompt) | CN (DeepSeek) | Distilled reasoning model, chain-of-thought |
| gemma3:1b | 1B | Ollama (raw prompt) | US (Google) | Sliding window attention architecture |
| phi4-mini:3.8b | 3.8B | Ollama (raw prompt) | US (Microsoft) | Structured reasoning, slightly above 3B tier |
| bitnet-3B | 3B | BitNet (llama-server) | US (Microsoft) | 1.58-bit base model, NOT instruction-tuned |
| bitnet-2B-4T | 2B | BitNet (llama-server) | US (Microsoft) | 1.58-bit, instruction-tuned on 4T tokens |

### Backend Details

Three inference backends were used:

- **Ollama (native tools):** Models that support Ollama's built-in `tools=` parameter in `ollama.chat()`. The API handles tool schema injection and structured output parsing natively.
- **Ollama (raw prompt):** Models that don't support Ollama's native tool API. Instead, a system prompt embedding the tool schemas is sent via `ollama.chat()` without `tools=`, and `<tool_call>{"name": ..., "arguments": ...}</tool_call>` tags are parsed from the plain-text response.
- **BitNet (llama-server):** Microsoft's bitnet.cpp `llama-server` running as a subprocess on port 8921. Same raw-prompt-and-parse approach as Ollama raw, but using the OpenAI-compatible `/v1/chat/completions` endpoint.

deepseek-r1:1.5b and gemma3:1b were initially tested with Ollama's native tool API but errored on every call (these models don't support it). phi4-mini:3.8b technically worked with native tools but only produced 1/9 tool calls -- switching to the raw prompt backend brought it to 6/9. All three were moved to the raw prompt backend for the final benchmark.

## Tools Available

Three mock tools were defined and provided to all models:

1. **get_weather**(city: string) -- returns mock weather data for a city
2. **search_files**(pattern: string) -- returns mock file search results for a glob pattern
3. **schedule_meeting**(title: string, time: string, attendees?: string[]) -- returns mock meeting confirmation

## Test Prompts

| ID | Prompt | Category | Expected Behavior |
|---|---|---|---|
| P1 | "What's the weather in Antwerp?" | Easy | Call `get_weather("Antwerp")` |
| P2 | "Find all Python files in the project." | Easy | Call `search_files("*.py")` |
| P3 | "Schedule a meeting called 'Sprint Review' for 2025-02-10T14:00:00 with alice@co.com and bob@co.com." | Easy (multi-arg) | Call `schedule_meeting(...)` with all params |
| P4 | "I'm heading to Brussels tomorrow, anything I should know?" | Ambiguous | Could call `get_weather("Brussels")` or decline |
| P5 | "What tools do you have access to?" | **Restraint** | Should NOT call any tool (meta question) |
| P6 | "What's the weather in the city where we have our next sprint review?" | Hard | Should call `get_weather` but city is unknown |
| P7 | "Oh hey, could you maybe like set up a meeting -- 'Q3 Roadmap' -- for next Tuesday at 3pm? I think dave@co.com and maybe susan@co.com should come" | Hard (noisy) | Call `schedule_meeting(...)`, extract params from noise |
| P8 | "Search for all files matching '*.py' and also tell me the weather in Paris." | Hard (dual-tool) | Call `search_files` and/or `get_weather` |
| P9 | "Can you write a Python script that checks the weather using an API?" | **Restraint** | Should NOT call any tool (code-writing request) |

## Scoring

**Agent Score** = (valid_tool_calls / 7) * 0.5 + (restraint / 2) * 0.5

- **Accuracy component (50%):** How many of the 7 clear tool-call prompts (P1, P2, P3, P4, P6, P7, P8) produced valid tool calls with parseable arguments.
- **Restraint component (50%):** How many of the 2 restraint prompts (P5, P9) were correctly left without a tool call.

Results are averaged across 3 runs using majority voting (tool_called if called in >50% of runs, tool_name by most-common, valid_args if any run produced valid args).

## Results

### Full Leaderboard (sorted by Agent Score)

| Rank | Model | Origin | Tool Calls | Valid Args | Avg Latency | Restraint | Agent Score |
|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:3b | CN | 6/9 | 6/6 | 3,861 ms | 2/2 | **0.929** |
| 1 | qwen2.5:0.5b | CN | 6/9 | 6/6 | 1,351 ms | 2/2 | **0.929** |
| 1 | smollm2:1.7b | US | 6/9 | 6/6 | 2,437 ms | 2/2 | **0.929** |
| 1 | phi4-mini:3.8b | US | 6/9 | 6/6 | 5,723 ms | 2/2 | **0.929** |
| 5 | ministral-3:3b | FR | 5/9 | 5/5 | 10,571 ms | 2/2 | **0.857** |
| 6 | qwen2.5:1.5b | CN | 4/9 | 4/4 | 3,126 ms | 2/2 | **0.786** |
| 7 | bitnet-2B-4T | US/1bit | 8/9 | 8/8 | 2,806 ms | 1/2 | **0.750** |
| 8 | llama3.2:3b | US | 9/9 | 9/9 | 2,786 ms | 0/2 | 0.500 |
| 8 | deepseek-r1:1.5b | CN | 0/9 | 0/0 | 7,535 ms | 2/2 | 0.500 |
| 8 | gemma3:1b | US | 0/9 | 0/0 | 4,139 ms | 2/2 | 0.500 |
| 8 | bitnet-3B | US/1bit | 0/9 | 0/0 | 16,046 ms | 2/2 | 0.500 |

### Edge Agent Mini Leaderboard (sub-2B models)

| Rank | Model | Origin | Tool Calls | Valid Args | Avg Latency | Restraint | Agent Score |
|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:0.5b | CN | 6/9 | 6/6 | 1,351 ms | 2/2 | **0.929** |
| 2 | smollm2:1.7b | US | 6/9 | 6/6 | 2,437 ms | 2/2 | **0.929** |
| 3 | qwen2.5:1.5b | CN | 4/9 | 4/4 | 3,126 ms | 2/2 | 0.786 |
| 4 | bitnet-2B-4T | US/1bit | 8/9 | 8/8 | 2,806 ms | 1/2 | 0.750 |
| 5 | deepseek-r1:1.5b | CN | 0/9 | 0/0 | 7,535 ms | 2/2 | 0.500 |
| 6 | gemma3:1b | US | 0/9 | 0/0 | 4,139 ms | 2/2 | 0.500 |

## Model-by-Model Analysis

### Tier 1: The 0.929 Club

**qwen2.5:0.5b (CN, 0.5B params) -- The Efficiency Champion**

At just 500 million parameters, this is the smallest model in the benchmark and the fastest by far (1,351 ms average). It nailed all three easy prompts, handled the noisy P7, and correctly identified both restraint prompts. It declined the ambiguous P4 and hard P6 -- conservative but never wrong. Perfect restraint, perfect accuracy on what it attempted. The fact that a model this small can reliably do structured tool-calling at sub-1.5s latency is remarkable.

**qwen2.5:3b (CN, 3B params) -- The Reliable Workhorse**

Same score as its 0.5B sibling but at 3x the latency. It gained the ambiguous P4 (calling get_weather for Brussels) which the smaller Qwen models declined. Lost P9 restraint in 2 of 3 runs (calling get_weather when asked to write code) but majority-vote saved it. Consistent across runs.

**smollm2:1.7b (US, 1.7B params) -- The HuggingFace Surprise**

HuggingFace's SmolLM2 continues to punch above its weight class. Perfect restraint (both P5 and P9 declined instantly in ~600ms -- the fastest "no" in the benchmark). Called all the right tools on easy and medium prompts. Only weakness: occasionally called the wrong tool on P8 (schedule_meeting instead of search_files in Run 2). At 2,437 ms average, it's a solid middle-ground between Qwen 0.5b's speed and the 3B models' capability.

**phi4-mini:3.8b (US, 3.8B params) -- The Redeemed Giant**

phi4-mini has the most dramatic story in this benchmark. In the initial run using Ollama's native tool API, it scored a dismal 1/9 tool calls (0.571 Agent Score). It understood prompts but refused to use the API. Switching to the raw prompt backend (system prompt + `<tool_call>` tag parsing) unlocked its full potential: 6/6 valid tool calls, perfect restraint, 0.929 Agent Score. Microsoft's reasoning training clearly works -- when given the right interface. Caveat: it's the slowest of the four leaders (5,723 ms average) and its P9 latency is wildly inconsistent (30s in Run 1, 4s in Run 2, 11s in Run 3).

### Tier 2: Strong but Flawed

**ministral-3:3b (FR, 3B params) -- EU Sovereignty Candidate, Slow but Accurate**

Mistral's 3B edge model is accurate (5/5 valid args, perfect restraint) but painfully slow. Average latency of 10,571 ms, with P4 and P9 taking 27-33 seconds each. The latency spikes happen on prompts where the model thinks hard before declining (P4 ambiguous, P9 restraint) -- it generates long text responses rather than quick refusals. When it does call tools, args are always valid. It missed P6 (multi-step reasoning) and P7 (noisy parameter extraction) -- the two hardest prompts. Functional for edge deployment where latency isn't critical.

**qwen2.5:1.5b (CN, 1.5B params) -- The Conservative Middle Child**

Scores lower than both its smaller (0.5b) and larger (3b) siblings. Perfect restraint but only 4/9 tool calls -- it declined the ambiguous P4, the hard P6, and the noisy P7 in 2 of 3 runs. It's the most cautious model in the benchmark: when unsure, it always declines rather than guessing. This is arguably the safest behavior for a production agent, but it costs accuracy points.

### Tier 3: Accuracy Without Restraint

**bitnet-2B-4T (US/1bit, 2B params) -- The BitNet Breakthrough**

See dedicated BitNet section below.

**llama3.2:3b (US, 3B params) -- The Tool-Call Maximalist**

9/9 tool calls, 9/9 valid arguments, 0/2 restraint. Llama 3.2 calls a tool on every single prompt, every single run, without exception. When asked "What tools do you have access to?" it calls `search_files`. When asked to write a Python script, it calls `search_files`. It's the most capable tool-caller in the benchmark (it even correctly handles P8's dual-tool request) but it has zero concept of when NOT to call a tool. This makes it dangerous as an autonomous agent -- it would execute actions on every input regardless of appropriateness. Agent Score of 0.500 reflects the 50/50 weighting: perfect accuracy, zero restraint.

### Tier 4: Non-Functional for Tool Calling

**deepseek-r1:1.5b (CN, 1.5B params) -- Thinks but Can't Act**

DeepSeek's distilled reasoning model understands what tools do -- its raw output shows responses like `get_weather(Antwerp)` and `search_files("*.py")` -- but it cannot produce the structured `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` format. It writes function-call-style text (`get_weather(Antwerp)`) instead of JSON. The chain-of-thought reasoning it was distilled from doesn't help with structured output format compliance. At 1.5B params, it apparently lacks the capacity to simultaneously reason about tool use AND follow an exact JSON schema. Average latency of 7,535 ms (slow for producing nothing useful) suggests it's spending tokens on think-chains that go nowhere. Not viable for tool calling at this size.

**gemma3:1b (US, 1B params) -- Correct Tags, Wrong Format**

Google's smallest instruction model gets tantalizingly close. Its raw output shows `<tool_call>get_weather(city: Antwerp)</tool_call>` -- it understood the system prompt, used the right tags, and identified the right tool with the right argument. But it used Python function-call syntax instead of JSON. The parser expects `{"name": "get_weather", "arguments": {"city": "Antwerp"}}` and gets `get_weather(city: Antwerp)`. In one run (Run 2), it even managed a valid `schedule_meeting` call for P7, suggesting the capability is there but unreliable. At 1B params, it can follow *most* of the schema but not the JSON serialization format. A custom parser for its function-call syntax could potentially recover these calls.

**bitnet-3B (US/1bit, 3B params) -- Base Model Gibberish**

The original BitNet 3B base model remains completely non-functional. It produces incoherent text fragments like "8.- the: ( with a eight the a to as, to a surr" for every prompt. This is expected -- it's a pre-training checkpoint without instruction tuning. Included as a control to demonstrate the instruction-tuning gap. Average latency of 16,046 ms (the slowest model in the benchmark) reflects the 3B parameter count running through BitNet's I2_S kernels without the optimization of the newer 2B-4T architecture.

## BitNet Deep Dive: 1.58-Bit Tool Calling

The most fascinating result in this benchmark is the BitNet-b1.58-2B-4T model. Microsoft's instruction-tuned 1.58-bit model represents a fundamentally different approach to neural network computation: every weight is constrained to {-1, 0, 1}, eliminating floating-point multiplication entirely.

### The Before and After

**bitnet-3B (base model):** Produces incoherent word salad for every prompt. Sample P1 output:

```
8.- the: ( with a eight the a to as, to a surr, as a a, said,
 all to a, the, with,,, with. how to,
 everything --
 to, --. -- the the. with.
```

**bitnet-2B-4T (instruction-tuned on 4T tokens):** Produces perfectly structured tool calls. Sample P1 output:

```
<tool_call>{"name": "get_weather", "arguments": {"city": "Antwerp"}}
```

This is the same 1.58-bit weight representation. The only difference is instruction tuning. The transformation from gibberish to structured JSON tool calls using only ternary weights is striking.

### What BitNet 2B-4T Gets Right

- **P1 (weather):** 3/3 runs produced identical, correct output: `get_weather(city: "Antwerp")`
- **P2 (file search):** 3/3 correct: `search_files(pattern: "*.py")`
- **P3 (meeting):** 3/3 correct: `schedule_meeting` with title, time, and attendees array
- **P6 (unknown city):** Called `get_weather` correctly but hallucinated a city ("New York" in runs 1 and 3, "San Francisco" in run 2). This is arguably the right *structure* -- it understood a weather tool was needed -- but fabricated the missing context.
- **P7 (noisy params):** Correctly extracted the meeting title, time reference, and attendee emails from informal language.
- **P8 (dual tool):** Emitted two sequential tool calls: `search_files(*.py)` followed by `get_weather(Paris)`. This is notable -- the model understood the prompt required two distinct tools and produced both, back-to-back, without a closing `</tool_call>` tag between them.

### Where BitNet 2B-4T Fails

- **P5 (restraint):** Called a tool in 2 of 3 runs. In Run 1, it invented `available_tools` (a non-existent tool). In Run 3, it called a tool named `tools`. It doesn't understand meta-questions about its own capabilities. This is the model's main weakness for agent use.
- **P9 (restraint):** Correctly declined in all 3 runs. It understood that "write a Python script" is a code-generation request, not a tool call. This asymmetric restraint (failing P5 but passing P9) suggests it can distinguish code-writing from tool-calling, but can't distinguish tool-listing from tool-calling.
- **P4 (ambiguous):** Called tools in all 3 runs but with inconsistent tool names: `search_files` in Run 1, `search_weather` (hallucinated) in Run 2, `search_ones` (hallucinated) in Run 3. The model knows a tool is appropriate but can't reliably pick the right one for ambiguous prompts.

### BitNet Latency Profile

BitNet 2B-4T has remarkably consistent latency across prompts:

| Prompt | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| P1 (weather) | 1,994 ms | 1,931 ms | 1,982 ms | 1,969 ms |
| P2 (files) | 1,719 ms | 1,625 ms | 1,801 ms | 1,715 ms |
| P3 (meeting) | 3,182 ms | 3,565 ms | 3,205 ms | 3,317 ms |
| P7 (noisy) | 2,588 ms | 3,093 ms | 3,102 ms | 2,928 ms |
| P8 (dual) | 2,518 ms | 2,416 ms | 2,617 ms | 2,517 ms |
| P9 (restraint) | 6,838 ms | 5,916 ms | 5,838 ms | 6,197 ms |

Simple prompts complete in ~2s. Multi-argument prompts (P3, P7) take ~3s. The restraint prompt P9 takes longest (~6s) because the model generates a full text response explaining why it won't call a tool. Compare this to the base bitnet-3B which averages 16-20s per prompt generating nonsense.

### What This Means

A model running entirely on ternary weights ({-1, 0, 1}) with no floating-point multiplication can:
- Parse natural language prompts and identify which tool to call
- Generate valid JSON with correct argument names and values
- Handle multi-argument functions (schedule_meeting with title, time, attendees)
- Emit multiple sequential tool calls for multi-tool requests
- Decline tool calls for code-writing requests (P9)

It cannot reliably:
- Distinguish meta-questions from action requests (P5)
- Handle ambiguous prompts without hallucinating tool names (P4)
- Control its eagerness to call tools (8/9 calls, only 1/2 restraint)

At 0.750 Agent Score and 2,806 ms average latency, BitNet 2B-4T is competitive with conventional 4-bit quantized models despite using 1.58-bit weights. It outscores llama3.2:3b (0.500) and qwen2.5:1.5b (0.786) in the edge tier.

## Failure Analysis

### Models That Failed to Call Tools

| Model | Failure Mode | Root Cause |
|---|---|---|
| deepseek-r1:1.5b | 0/9 tools across 27 calls | Outputs function-call syntax (`get_weather(Antwerp)`) instead of JSON. Chain-of-thought distillation doesn't teach structured output formatting. |
| gemma3:1b | 0/9 tools (averaged) | Outputs `<tool_call>get_weather(city: Antwerp)</tool_call>` -- correct tags, wrong inner format. Uses Python kwargs syntax instead of JSON. 1B params insufficient for exact schema compliance. |
| bitnet-3B | 0/9 tools across 27 calls | Base model without instruction tuning. Produces incoherent token sequences. Not a capability failure -- simply untrained for the task. |

### Models That Failed on Restraint

| Model | P5 (meta) | P9 (code) | Failure Mode |
|---|---|---|---|
| llama3.2:3b | FAIL (3/3) | FAIL (3/3) | Calls a tool on every prompt without exception. Zero restraint capability. Called `get_weather` for P5 and `search_files` for P9. |
| bitnet-2B-4T | FAIL (2/3) | PASS (3/3) | Invented non-existent tools (`available_tools`, `tools`) for the meta-question. Correctly declined P9. Asymmetric restraint. |
| qwen2.5:3b | PASS (3/3) | FAIL (2/3) | Called `get_weather` on P9 in Runs 1 and 2 (keyword-triggered by "weather" in the prompt). Majority-vote tiebreaker saved the averaged result. |

### Per-Prompt Difficulty Analysis

| Prompt | Models Passing (of 11) | Hardest For |
|---|---|---|
| P1 (easy weather) | 7 | deepseek-r1, gemma3, bitnet-3B, bitnet-2B-4T all pass structurally |
| P2 (easy files) | 7 | Same non-functional models fail |
| P3 (easy meeting) | 7 | Same pattern |
| P4 (ambiguous) | 4 | Most models decline (conservative). qwen2.5:3b, llama3.2, ministral, phi4-mini call tools |
| P5 (restraint) | 8 | llama3.2, bitnet-2B-4T fail. qwen2.5:3b barely passes |
| P6 (hard context) | 4 | Requires inferring missing context. qwen2.5:0.5b, smollm2, phi4-mini (Run 3), bitnet-2B-4T call tools |
| P7 (noisy params) | 6 | ministral-3:3b, qwen2.5:1.5b decline. Parameter extraction from informal text is hard |
| P8 (dual tool) | 7 | All functional models handle this. Only first tool is captured by parser |
| P9 (restraint) | 9 | Only llama3.2:3b and bitnet-2B-4T (in some runs) fail. Most models recognize code-writing requests |

## Latency Comparison

Average latency per model across all 9 prompts, 3 runs:

| Model | Avg Latency | Notes |
|---|---|---|
| qwen2.5:0.5b | 1,351 ms | Fastest overall. Sub-1s on simple prompts |
| smollm2:1.7b | 2,437 ms | Fast refusals (~600ms on restraint prompts) |
| llama3.2:3b | 2,786 ms | Consistent, no thinking pauses |
| bitnet-2B-4T | 2,806 ms | Remarkably fast for 1.58-bit inference |
| qwen2.5:1.5b | 3,126 ms | Moderate |
| qwen2.5:3b | 3,861 ms | Moderate |
| gemma3:1b | 4,139 ms | P9 spikes to 20-26s (generates long code) |
| phi4-mini:3.8b | 5,723 ms | High variance. P9 ranges from 4s to 30s |
| deepseek-r1:1.5b | 7,535 ms | Chain-of-thought overhead for no usable output |
| ministral-3:3b | 10,571 ms | P4 and P9 take 27-33s (long text generation) |
| bitnet-3B | 16,046 ms | Slowest. Generating incoherent tokens is expensive |

## Conclusions

1. **Tool calling works at 500M parameters.** qwen2.5:0.5b ties for the top score at 1.35s average latency. There is no inherent floor for tool-calling capability -- training data and instruction tuning matter more than parameter count.

2. **Restraint is the differentiator.** The gap between 0.929 and 0.500 is entirely restraint. llama3.2:3b proves you can have perfect accuracy and still be a bad agent if you call tools on everything.

3. **1.58-bit weights can do structured tool calling.** BitNet-2B-4T is the proof point. Ternary weights ({-1, 0, 1}) are sufficient for JSON generation, argument extraction, and multi-tool dispatch. The model's main weakness is restraint (P5), not accuracy.

4. **The right backend matters as much as the model.** phi4-mini went from 0.571 to 0.929 just by switching from Ollama's native tool API to a raw system prompt. Models have different strengths; the interface should match.

5. **Chain-of-thought doesn't help tool calling at small scale.** deepseek-r1:1.5b's reasoning distillation produced a model that thinks about tools but can't format tool calls. At 1.5B params, the reasoning overhead crowds out format compliance.

6. **Google and DeepSeek's smallest models can't do structured JSON.** Both understand the concept of tool calling but produce non-JSON formats (function-call syntax, kwargs). This appears to be a hard capability boundary somewhere between 1B and 1.5B params for non-Qwen architectures.
