# Local LLM Tool-Calling Benchmark Report (Round 4)

**Date:** 2026-02-07
**Runs:** 3 per model/prompt combination (396 total inference calls)
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

deepseek-r1:1.5b and gemma3:1b were initially tested with Ollama's native tool API but errored on every call (these models don't support it). phi4-mini:3.8b technically worked with native tools but produced far fewer tool calls -- switching to the raw prompt backend dramatically improved its output. All three were moved to the raw prompt backend for the final benchmark.

> **Methodology note:** This benchmark evaluates model-protocol pairs rather than models in isolation. Tool-calling behavior is strongly dependent on the interaction contract (native API vs. raw prompt vs. OpenAI-compatible endpoint). The backend/mode columns in the leaderboard exist to make this dependency explicit.

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
| P10 | "I have a meeting with a client in Bruges next Thursday. Should I take the train or cycle?" | **Hard (implicit reasoning)** | Call `get_weather("Bruges")` -- transport choice depends on weather |
| P11 | "Don't check the weather in Antwerp, just find me the quarterly report." | **Hard (negation)** | Call `search_files("quarterly report")` -- explicit negation of weather |
| P12 | "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?" | **Hard (context awareness)** | Call `schedule_meeting(...)` -- weather already provided, action needed |

### What Changed from Round 3

Round 4 adds three "hard" prompts (P10-P12) designed to test whether models can pick the *right* tool when misleading keywords are present. These prompts broke the Round 3 plateau where four models tied at 0.929.

- **P10** mentions a meeting but the correct tool is `get_weather` (transport choice depends on weather). Calling `schedule_meeting` is the worst wrong answer -- the meeting already exists.
- **P11** explicitly says "don't check the weather" but mentions Antwerp. The correct tool is `search_files`. Calling `get_weather` means the model ignored a direct instruction.
- **P12** provides the weather in the prompt itself. The correct action is `schedule_meeting`. Calling `get_weather` means the model didn't read the context.

## Scoring

Five metrics capture independent capabilities:

- **Action Score** = correct_tool_calls / 10. How many of the 10 actionable prompts (P1-P4, P6-P8, P10-P12) produced valid tool calls with the correct tool. For P10-P12, the tool must match the expected tool to count as correct. Measures execution capability.
- **Restraint Score** = correct_refusals / 2. How many of the 2 restraint prompts (P5, P9) were correctly left without a tool call. Measures policy calibration.
- **Wrong Tool** = count of specifically-bad tool calls on P10-P12. Each hard prompt has a "wrong tool" that is worse than not calling any tool at all (e.g., calling `get_weather` on P11 when explicitly told "don't check the weather"). Range: 0-3. Measures judgment under misleading context.
- **Reliability** = average per-prompt (successful_runs / 3). Computed from per-run data before majority voting. For P10-P12, "successful" means calling the correct expected tool. Measures deployability.
- **Multi-Tool Accuracy** = correct_tools / required_tools for P8 only. P8 requires both `search_files` and `get_weather`. Ollama's native tool API returns only the first tool call, so this metric is N/A for native-tools models.
- **Agent Score** = Action × 0.4 + Restraint × 0.3 + Wrong-Tool-Avoidance × 0.3. Where Wrong-Tool-Avoidance = (3 - wrong_tool_count) / 3. The three-part composite weights execution (40%), policy calibration (30%), and judgment (30%).

> The Agent Score formula changed from Round 3's `Action × 0.5 + Restraint × 0.5` to incorporate wrong-tool penalties. A model that calls tools aggressively but picks the wrong ones is now penalized. A model that conservatively declines uncertain prompts is rewarded for avoiding wrong tools.

> Majority voting reflects correctness. Reliability reflects deployability. Agents fail in production from rare errors, not average ones.

Results are averaged across 3 runs using majority voting (tool_called if called in >50% of runs, tool_name by most-common, valid_args if any run produced valid args). Reliability is computed before majority voting to capture per-run instability.

**Note on P4:** P4 ("I'm heading to Brussels tomorrow, anything I should know?") is ambiguous. Calling `get_weather("Brussels")` is reasonable, and declining is also reasonable. Calling any tool with valid arguments counts as correct for the Action Score.

## Results

### Full Leaderboard (sorted by Agent Score)

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Reliability | Multi-Tool | Agent Score |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.500 | 1.000 | 0 | 0.611 | N/A* | **0.800** |
| 1 | ministral-3:3b | Ollama | native-tools | FR | 0.500 | 1.000 | 0 | 0.583 | N/A* | **0.800** |
| 3 | phi4-mini:3.8b | Ollama | raw-schema | US | 0.700 | 1.000 | 2 | 0.750 | 0.500 | **0.680** |
| 4 | qwen2.5:3b | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.722 | N/A* | 0.670 |
| 5 | llama3.2:3b | Ollama | native-tools | US | 0.900 | 0.000 | 0 | 0.722 | N/A* | 0.660 |
| 6 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.667 | N/A* | 0.640 |
| 6 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.694 | N/A* | 0.640 |
| 8 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 |
| 8 | gemma3:1b | Ollama | raw-schema | US | 0.000 | 1.000 | 0 | 0.194 | 0.000 | 0.600 |
| 8 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 |
| 11 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.800 | 0.500 | 2 | 0.750 | 1.000 | 0.570 |

\*Ollama native-tools API returns only the first tool call.

### Edge Agent Mini Leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Reliability | Multi-Tool | Agent Score |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.500 | 1.000 | 0 | 0.611 | N/A* | **0.800** |
| 2 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.667 | N/A* | 0.640 |
| 2 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.694 | N/A* | 0.640 |
| 4 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 |
| 4 | gemma3:1b | Ollama | raw-schema | US | 0.000 | 1.000 | 0 | 0.194 | 0.000 | 0.600 |
| 6 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.800 | 0.500 | 2 | 0.750 | 1.000 | 0.570 |

### Hard Prompts P10-P12 (detailed)

| Model | P10 Tool | P10 | P11 Tool | P11 | P12 Tool | P12 | Wrong |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b | get_weather | OK | search_files | OK | get_weather | WRONG | 1 |
| qwen2.5:1.5b | (none) | miss | (none) | miss | schedule_meeting | OK | 0 |
| qwen2.5:0.5b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| llama3.2:3b | search_files | wrong? | search_files | OK | schedule_meeting | OK | 0 |
| smollm2:1.7b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| ministral-3:3b | (none) | miss | (none) | miss | (none) | miss | 0 |
| deepseek-r1:1.5b | (none) | miss | (none) | miss | (none) | miss | 0 |
| gemma3:1b | (none) | miss | (none) | miss | (none) | miss | 0 |
| phi4-mini:3.8b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| bitnet-3B | (none) | miss | (none) | miss | (none) | miss | 0 |
| bitnet-2B-4T | schedule_meeting | WRONG | search_files | OK | get_weather | WRONG | 2 |

**Legend:** "OK" = correct tool. "WRONG" = called the specifically-bad tool (penalized in Wrong Tool metric). "wrong?" = wrong tool but not the worst choice (not penalized). "miss" = didn't call any tool (no penalty, but no Action credit).

## Model-by-Model Analysis

### Tier 1: Conservative and Correct (Agent Score 0.800)

**qwen2.5:1.5b (CN, 1.5B params) -- The Conservative Winner**

The most cautious model in the benchmark rose from 6th place (Round 3) to joint 1st. Perfect restraint (both P5 and P9 declined), zero wrong tool calls, and the only model to correctly call `schedule_meeting` on P12 (the hardest prompt). It declined P10 and P11 rather than guessing -- losing Action points but avoiding wrong-tool penalties. This is the ideal agent behavior: when uncertain, don't act. At 2,582 ms average latency, it's practical for real deployment. The trade-off is clear: Action 0.500 means it only handles half the actionable prompts, but everything it does attempt is correct.

**ministral-3:3b (FR, 3B params) -- EU Sovereignty Candidate, Now a Leader**

Mistral's 3B edge model matched qwen2.5:1.5b's Agent Score through the same strategy: perfect restraint, zero wrong tools, conservative declining. It passed P1-P3 and P4 (calling get_weather for Brussels), and P8 (search_files). It missed P6 (unknown city), P7 (noisy params), and all three hard prompts. Its conservatism on P10-P12 -- declining rather than guessing -- is exactly what kept its score high. The cost: 8,252 ms average latency, with P4 and P9 taking 19-33 seconds as the model generates long text responses before declining.

### Tier 2: Capable but Penalized (Agent Score 0.660-0.680)

**phi4-mini:3.8b (US, 3.8B params) -- Execution Meets Bad Judgment**

phi4-mini has the third-highest Action Score (0.700) and perfect Restraint, but 2 wrong tool calls on P11 and P12 drag it to 0.680. It called `get_weather` on P11 despite being told "don't check the weather" -- a direct negation failure. It called `get_weather` on P12 when the weather was already provided in the prompt -- a context-reading failure. These are the same keyword-matching failures that affect qwen2.5:0.5b and smollm2:1.7b. At 3.8B parameters, phi4-mini should be the most capable model in the benchmark, but its hard-prompt judgment is no better than models one-seventh its size. Reliability of 0.750 (down from 0.926 in Round 3) confirms the hard prompts exposed genuine weaknesses.

**qwen2.5:3b (CN, 3B params) -- The Fallen Leader**

The most dramatic fall from Round 3. Joint 1st (0.929) to 4th (0.670). Two problems: it failed P9 restraint (calling `get_weather` when asked to write code -- keyword-triggered by "weather" in the prompt), and it called `get_weather` on P12 (weather already provided). The P9 failure drops Restraint to 0.500; the P12 failure adds 1 wrong tool. Combined: 0.800 × 0.4 + 0.500 × 0.3 + 0.667 × 0.3 = 0.670. qwen2.5:3b is now outperformed by its 1.5B sibling, which avoided both mistakes through conservatism. The lesson: being trigger-happy on tools is more costly under the new scoring than being conservative.

**llama3.2:3b (US, 3B params) -- Tool-Call Maximalist, Redeemed by Judgment**

Still calls a tool on every prompt (0/2 restraint), but the hard prompts revealed a surprise: llama3.2 has decent tool selection. It correctly called `search_files` for P11 ("find the quarterly report") and `schedule_meeting` for P12 ("schedule an indoor meeting"). For P10 it called `search_files` -- wrong, but not `schedule_meeting` (the penalized wrong tool), so no Wrong Tool penalty. Zero restraint still caps its score, but 0.660 is a meaningful improvement over Round 3's 0.500, thanks to the new formula giving 0.3 weight to wrong-tool-avoidance (where it scored perfectly). Its Action Score of 0.900 is the highest in the benchmark.

### Tier 3: Keyword-Trapped (Agent Score 0.640)

**qwen2.5:0.5b (CN, 0.5B params) -- Speed Champion, Judgment Victim**

The smallest and fastest model (874 ms average) dropped from joint 1st (Round 3) to 6th. It called `get_weather` on both P11 (told not to) and P12 (weather already given) -- pure keyword matching on "weather." At 0.5B parameters, the model lacks the capacity to parse negation or notice that information is already provided. Its Action Score (0.600) reflects both the P10-P12 misses and the wrong tool calls. Still remarkable for basic tool calling at sub-1s latency, but the hard prompts exposed a hard capability floor.

**smollm2:1.7b (US, 1.7B params) -- Same Keyword Trap, Bigger Model**

Identical failure pattern to qwen2.5:0.5b: called `get_weather` on P11 and P12, both keyword-triggered. At 1.7B parameters it should do better than the 0.5B Qwen, but the hard prompts show the same weakness. Perfect restraint on P5 and P9 (fast "no" in ~600ms), but no ability to resist "weather" as a keyword trigger when it appears alongside other tools. Reliability of 0.694 is higher than qwen2.5:0.5b's 0.667, reflecting slightly more consistency on the prompts it does handle correctly.

### Tier 4: Non-Functional for Tool Calling (Agent Score 0.600)

**deepseek-r1:1.5b (CN, 1.5B params) -- Thinks but Can't Act**

DeepSeek's distilled reasoning model understands what tools do -- its raw output shows responses like `get_weather(Antwerp)` and `search_files("*.py")` -- but it cannot produce the structured `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` format. It writes function-call-style text instead of JSON. At 5,758 ms average latency, it's slow for producing nothing useful. Not viable for tool calling at this size.

**gemma3:1b (US, 1B params) -- Correct Tags, Wrong Format**

Google's smallest instruction model outputs `<tool_call>get_weather(city: Antwerp)</tool_call>` -- correct tags, right tool, right argument, but Python function-call syntax instead of JSON. At 1B params, it can follow most of the schema but not the JSON serialization format. A custom parser for its syntax could potentially recover these calls. Average latency of 2,543 ms with P9 spiking to 15-21s when it generates long code responses.

**bitnet-3B (US/1bit, 3B params) -- Base Model Gibberish**

The original BitNet 3B base model remains completely non-functional. Produces incoherent text fragments for every prompt. This is expected -- it's a pre-training checkpoint without instruction tuning. Included as a control. Average latency of 15,498 ms (the slowest model in the benchmark).

### Tier 5: Strong Actuator, Weak Policy (Agent Score 0.570)

**bitnet-2B-4T (US/1bit, 2B params) -- Execution Without Judgment**

BitNet 2B-4T fell from 7th (0.750) to last among functional models (0.570). Its Action Score remains high (0.800) and it's still the only model to achieve Multi-Tool 1.000 on P8 (emitting both `search_files` and `get_weather` back-to-back). But the hard prompts were devastating: it called `schedule_meeting` for P10 (should be get_weather -- WRONG) and `get_weather` for P12 (weather already provided -- WRONG). Combined with P5 restraint failure, it now has Restraint 0.500 and Wrong Tool 2. The model is a strong actuator with weak judgment -- it knows *how* to call tools but not *which* tool to call under ambiguity.

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

This is the same 1.58-bit weight representation. The only difference is instruction tuning.

### What BitNet 2B-4T Gets Right

- **P1 (weather):** 3/3 runs produced identical, correct output: `get_weather(city: "Antwerp")`
- **P2 (file search):** 3/3 correct: `search_files(pattern: "*.py")`
- **P3 (meeting):** 3/3 correct: `schedule_meeting` with title, time, and attendees array
- **P6 (unknown city):** Called `get_weather` correctly but hallucinated a city ("New York" in all 3 runs). Right structure, fabricated context.
- **P7 (noisy params):** Correctly extracted the meeting title, time reference, and attendee emails from informal language.
- **P8 (dual tool):** Emitted two sequential tool calls: `search_files(*.py)` followed by `get_weather(Paris)`. 3/3 consistent. The only model to achieve Multi-Tool Accuracy 1.000.
- **P11 (negation):** Correctly called `search_files` in 2 of 3 runs, resisting the "weather" keyword trap that caught 4 other models. In Run 3 it declined entirely ("I'm sorry, but I can't fulfill that request").

### Where BitNet 2B-4T Fails

- **P5 (restraint):** Called a tool in 2 of 3 runs. In Run 1, it invented `available_tools` (a non-existent tool). It doesn't understand meta-questions about its own capabilities.
- **P10 (implicit reasoning):** Called `schedule_meeting` in all 3 runs -- the specifically penalized wrong tool. The meeting already exists; the question is about weather for transport. The model saw "meeting" and "client" and pattern-matched to `schedule_meeting`.
- **P12 (context awareness):** Called `get_weather` in 2 of 3 runs despite the weather being provided in the prompt. In Run 2, it actually produced a correct `schedule_meeting` response with reasoning ("It seems like you are asking for a suggestion to schedule an indoor meeting..."), showing the capability exists but isn't consistent.
- **P4 (ambiguous):** Called `search_files` in all 3 runs for a travel question -- the tool choice is odd but not penalized since P4 is ambiguous.

### BitNet Latency Profile

BitNet 2B-4T has remarkably consistent latency across prompts:

| Prompt | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| P1 (weather) | 1,904 ms | 1,380 ms | 1,358 ms | 1,547 ms |
| P2 (files) | 1,604 ms | 1,272 ms | 1,348 ms | 1,408 ms |
| P3 (meeting) | 2,875 ms | 2,130 ms | 2,585 ms | 2,530 ms |
| P7 (noisy) | 2,944 ms | 2,436 ms | 2,579 ms | 2,653 ms |
| P8 (dual) | 2,314 ms | 1,980 ms | 1,979 ms | 2,091 ms |
| P9 (restraint) | 6,250 ms | 7,133 ms | 6,326 ms | 6,570 ms |

Simple prompts complete in ~1.5s. Multi-argument prompts (P3, P7) take ~2.5s. The restraint prompt P9 takes longest (~6.5s) because the model generates a full text response. Compare this to the base bitnet-3B which averages 15-18s per prompt generating nonsense.

### What This Means

A model running entirely on ternary weights ({-1, 0, 1}) with no floating-point multiplication can:
- Parse natural language prompts and identify which tool to call
- Generate valid JSON with correct argument names and values
- Handle multi-argument functions (schedule_meeting with title, time, attendees)
- Emit multiple sequential tool calls for multi-tool requests (the only model to do so)
- Decline tool calls for code-writing requests (P9)
- Resist keyword traps when given explicit negation (P11, 2/3 runs)

It cannot reliably:
- Distinguish meta-questions from action requests (P5)
- Reason about implicit context (P10: "should I cycle" requires weather, not meeting scheduling)
- Notice when information is already provided (P12: weather given in prompt)

At 0.570 Agent Score and 2,340 ms average latency, BitNet 2B-4T's ranking dropped from Round 3 due to the hard prompts exposing its judgment weaknesses. But its raw execution capability remains impressive: Action 0.800 and Multi-Tool 1.000 using only ternary weights.

## Failure Analysis

### The Wrong-Tool Trap

P10-P12 revealed a consistent failure pattern across our 3-run sample: keyword-triggered wrong tool calls. Five of eight functional models committed at least one wrong tool call:

| Model | P11 (negation) | P12 (context) | Pattern |
|---|---|---|---|
| qwen2.5:0.5b | get_weather WRONG | get_weather WRONG | Keyword "weather" overrides all context |
| smollm2:1.7b | get_weather WRONG | get_weather WRONG | Same pattern |
| phi4-mini:3.8b | get_weather WRONG | get_weather WRONG | Same pattern at 3.8B params |
| qwen2.5:3b | search_files OK | get_weather WRONG | Resisted P11 negation, failed P12 context |
| bitnet-2B-4T | search_files OK | get_weather WRONG | Resisted P11 negation, failed P12 context |

The pattern is consistent across runs: "weather" in the prompt appears to trigger `get_weather` regardless of context. P11's explicit "don't check the weather" was ignored by 3 models. P12's "the weather is 8°C and rainy" (information already provided) triggered a redundant weather check in 5 models. Only qwen2.5:1.5b, ministral-3:3b, and llama3.2:3b avoided all wrong-tool penalties.

### Models That Failed to Call Tools

| Model | Failure Mode | Root Cause |
|---|---|---|
| deepseek-r1:1.5b | 0/12 tools across 36 calls | Outputs function-call syntax (`get_weather(Antwerp)`) instead of JSON. Chain-of-thought distillation doesn't teach structured output formatting. |
| gemma3:1b | 0/12 tools (averaged) | Outputs `<tool_call>get_weather(city: Antwerp)</tool_call>` -- correct tags, wrong inner format. Uses Python kwargs syntax instead of JSON. |
| bitnet-3B | 0/12 tools across 36 calls | Base model without instruction tuning. Produces incoherent token sequences. |

### Models That Failed on Restraint

| Model | P5 (meta) | P9 (code) | Failure Mode |
|---|---|---|---|
| llama3.2:3b | FAIL (3/3) | FAIL (3/3) | Calls a tool on every prompt without exception. Zero restraint capability. |
| qwen2.5:3b | PASS (3/3) | FAIL (majority) | Called `get_weather` on P9, keyword-triggered by "weather" in the prompt. |
| bitnet-2B-4T | FAIL (2/3) | PASS (3/3) | Invented non-existent tools for the meta-question. Correctly declined P9. |

### Per-Prompt Difficulty Analysis

| Prompt | Models Correct (of 11) | Notes |
|---|---|---|
| P1 (easy weather) | 8 | All functional models pass |
| P2 (easy files) | 8 | Same |
| P3 (easy meeting) | 8 | Same |
| P4 (ambiguous) | 4 | Conservative models decline. qwen2.5:3b, llama3.2, ministral, phi4-mini call tools |
| P5 (restraint) | 8 | llama3.2 and bitnet-2B-4T fail |
| P6 (hard context) | 5 | Requires inferring missing context. qwen2.5:0.5b, smollm2, phi4-mini, bitnet-2B-4T call tools |
| P7 (noisy params) | 6 | ministral-3:3b, qwen2.5:1.5b decline |
| P8 (dual tool) | 8 | All functional models handle this (first tool captured) |
| P9 (restraint) | 8 | llama3.2 and qwen2.5:3b fail |
| P10 (implicit reasoning) | 1 | Only qwen2.5:3b called the correct tool. Hardest prompt by correct-tool count |
| P11 (negation) | 3 | qwen2.5:3b, llama3.2, bitnet-2B-4T resisted the keyword trap |
| P12 (context awareness) | 2 | Only qwen2.5:1.5b and llama3.2 called schedule_meeting |

P10 is the hardest prompt in the benchmark: only 1 model called the correct tool (`get_weather`). P12 is close behind at 2 correct. The hard prompts have dramatically better discriminative power than P1-P9.

## Latency Comparison

Average latency per model across all 12 prompts, 3 runs:

| Model | Avg Latency | Notes |
|---|---|---|
| qwen2.5:0.5b | 874 ms | Fastest overall. Sub-500ms on simple prompts |
| smollm2:1.7b | 2,064 ms | Fast refusals (~600ms on restraint prompts) |
| llama3.2:3b | 2,152 ms | Consistent, no thinking pauses |
| bitnet-2B-4T | 2,340 ms | Remarkably fast for 1.58-bit inference |
| gemma3:1b | 2,543 ms | P9 spikes to 15-21s (generates long code) |
| qwen2.5:1.5b | 2,582 ms | Moderate |
| qwen2.5:3b | 3,543 ms | Moderate |
| phi4-mini:3.8b | 5,661 ms | High variance. P11 took 19s in one run |
| deepseek-r1:1.5b | 5,758 ms | Chain-of-thought overhead for no usable output |
| ministral-3:3b | 8,252 ms | P4 takes 19-29s (long text generation before declining) |
| bitnet-3B | 15,498 ms | Slowest. Generating incoherent tokens is expensive |

## Conclusions

1. **Hard prompts broke the plateau.** In Round 3, four models tied at 0.929. P10-P12 spread them from 0.640 to 0.800, with two new leaders (qwen2.5:1.5b, ministral-3:3b) that weren't in the top group before. The Round 3 ceiling was an artifact of the benchmark not being hard enough.

2. **Not calling a tool is better than calling the wrong one.** The new scoring formula (Action × 0.4 + Restraint × 0.3 + Wrong-Tool-Avoidance × 0.3) rewards conservative models. qwen2.5:1.5b and ministral-3:3b scored highest by declining uncertain prompts rather than guessing wrong. This matches real-world agent requirements: an agent that does nothing when confused is safer than one that takes the wrong action.

3. **Keyword matching appears to be a common failure pattern for small models.** Five of eight functional models called `get_weather` when they saw "weather" in the prompt, regardless of context. P11 ("Don't check the weather") and P12 ("The weather is 8°C and rainy") both triggered reflexive tool calls. This is consistent with sub-4B models relying heavily on keyword matching rather than semantic understanding for tool selection, though confirming this would require a larger prompt set.

4. **Bigger isn't always better within a model family.** qwen2.5:1.5b (0.800) now outperforms qwen2.5:3b (0.670). The 3B model's aggression -- calling `get_weather` for P9 and P12 -- is more costly than the 1.5B model's conservatism. The relationship between parameter count and agent quality is non-monotonic when judgment is measured. This ranking is sensitive to the scoring weights: the formula gives 60% combined weight to restraint and wrong-tool-avoidance, which structurally favors conservative models. Under an action-heavy formula the 3B model would rank higher. The underlying observation — that the larger model makes more wrong calls while the smaller model declines more — is robust; which behavior is "better" depends on the deployment context.

5. **Agent behavior separates into four independent capabilities.** Execution (Action), policy calibration (Restraint), judgment (Wrong Tool), and stability (Reliability). These are weakly correlated. BitNet 2B-4T has Action 0.800 but Wrong Tool 2. llama3.2:3b has Action 0.900 but Restraint 0.000. qwen2.5:1.5b has Action 0.500 but perfect on everything else. No single model excels on all four.

6. **A 2B model with 1.58-bit weights achieves strong execution but weak judgment on hard prompts.** BitNet 2B-4T retained strong execution (Action 0.800, Multi-Tool 1.000) but failed the judgment tests (Wrong Tool 2, Restraint 0.500). Whether the judgment failures stem from the ternary weight representation, the 2B parameter count, or the training data composition (which differs from the Ollama models) can't be determined from this benchmark alone — isolating the cause would require a 2B model with identical training but conventional weights as a control.

7. **P12 is the single best discriminator prompt.** "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?" Only 2 of 11 models correctly called `schedule_meeting`. It requires reading the provided context (weather is known), ignoring the keyword trigger ("weather"), and identifying the actual action requested (scheduling). This tests three capabilities simultaneously: context awareness, keyword resistance, and action identification.

8. **This benchmark evaluates model-protocol pairs, not models in isolation.** phi4-mini uses raw-schema while qwen2.5:3b uses native-tools. The backend/mode columns in the leaderboard exist to make this dependency explicit.
