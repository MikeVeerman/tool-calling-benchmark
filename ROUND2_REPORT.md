# Local LLM Tool-Calling Benchmark Report (Round 2)

**Date:** 2026-02-09
**Models:** 21 (11 original + 10 community-requested)
**Runs:** 3 per model/prompt combination (756 total inference calls)
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

## What Changed from Round 1

Round 2 uses the same benchmark design as Round 1 (same 12 prompts, same scoring formula, same test harness). The only changes are:

1. **10 new models** added from community requests on [the Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1qyg10z/) (163 upvotes, 68 comments).
2. **All models rerun fresh** (3 runs each). Original model scores may differ from Round 1 due to run-to-run variance. See [Score Changes for Original Models](#score-changes-for-original-models) for details.
3. **New backend: llama.cpp** added for models not available through Ollama (LFM2.5).
4. **`think=False` added** to the raw-schema backend to handle thinking-mode models (Qwen3-based) that otherwise produce empty content fields.

## New Models Tested

| Model | Params | Backend | Origin | Requested By | Notes |
|---|---|---|---|---|---|
| qwen3:0.6b | 0.6B | Ollama (native tools) | CN (Alibaba) | u/Far-Low-4705, u/noctrex, +4 others | Thinking-capable, smallest Qwen3 |
| qwen3:1.7b | 1.7B | Ollama (native tools) | CN (Alibaba) | u/Far-Low-4705 | Thinking-capable |
| qwen3:4b | 4B | Ollama (native tools) | CN (Alibaba) | u/JsThiago5 | Thinking-capable, longest latency |
| functiongemma | 270M | Ollama (native tools) | US (Google) | u/HankyHanks, u/Far-Low-4705 | Fine-tuned specifically for function calling |
| granite3.3:2b | 2B | Ollama (native tools) | US (IBM) | -- | IBM's earlier edge model |
| granite4:3b | 3B | Ollama (native tools) | US (IBM) | u/novocast | IBM's latest Granite generation |
| llama3.2:1b | 1B | Ollama (native tools) | US (Meta) | -- | Smallest Llama 3.2 |
| lfm2.5:1.2b | 1.2B | llama.cpp (raw prompt) | US (Liquid AI) | u/noctrex, u/Selfdrivinggolfcart, u/RnRau | State-space hybrid architecture |
| smollm3:3b | 3B | Ollama (raw prompt) | US (HuggingFace) | u/vasileer | SmolLM2 successor, thinking-capable |
| jan-v3:4b | 4B | Ollama (raw prompt) | US (jan.ai) | u/DataGOGO, u/IAmBobC | Qwen3 fine-tune with thinking |

### Models Attempted but Not Included

| Model | Reason |
|---|---|
| DeepBrainz-R1-2B | Community GGUF (mradermacher) outputs Thai/garbage text. Model appears broken at the quantization level. |
| Gemma 3n (e2b) | 5.6 GB download, exceeds the scope of this small-model benchmark. |

### Backend Details

Four inference backends were used:

- **Ollama (native tools):** Models that support Ollama's built-in `tools=` parameter. The API handles tool schema injection and structured output parsing natively.
- **Ollama (raw prompt):** Models that don't support Ollama's native tool API. A system prompt embedding the tool schemas is sent via `ollama.chat()` without `tools=`, and `<tool_call>{"name": ..., "arguments": ...}</tool_call>` tags are parsed from the plain-text response.
- **llama.cpp (raw prompt):** Stock llama.cpp `llama-server` running as a subprocess. Same raw-prompt-and-parse approach, using the OpenAI-compatible `/v1/chat/completions` endpoint. Used for models not available through Ollama.
- **BitNet (llama-server):** Microsoft's bitnet.cpp `llama-server`. Same raw-prompt approach on port 8921.

New in Round 2: smollm3:3b doesn't support Ollama's native tool API (returns HTTP 400) and was moved to raw prompt. jan-v3:4b supports the native API but produces empty content when thinking is enabled; it was moved to raw prompt with `think=False` to get usable output.

## Scoring

Unchanged from Round 1. Five metrics capture independent capabilities:

- **Action Score** = correct_tool_calls / 10. How many of the 10 actionable prompts (P1-P4, P6-P8, P10-P12) produced valid tool calls with the correct tool.
- **Restraint Score** = correct_refusals / 2. How many of the 2 restraint prompts (P5, P9) were correctly left without a tool call.
- **Wrong Tool** = count of specifically-bad tool calls on P10-P12 (range: 0-3).
- **Reliability** = average per-prompt (successful_runs / 3), computed before majority voting.
- **Multi-Tool Accuracy** = correct_tools / required_tools for P8 only. N/A for native-tools models (Ollama returns only the first tool call).
- **Agent Score** = Action x 0.4 + Restraint x 0.3 + Wrong-Tool-Avoidance x 0.3, where Wrong-Tool-Avoidance = (3 - wrong_tool_count) / 3.

Results averaged across 3 runs using majority voting.

## Results

### Full Leaderboard (sorted by Agent Score)

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Reliability | Multi-Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **qwen3:0.6b** | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | 0.750 | N/A* | **0.880** | 3,645 |
| **1** | **qwen3:4b** | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | 0.750 | N/A* | **0.880** | 63,717 |
| 3 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 0 | 0.639 | N/A* | 0.840 | 2,211 |
| 4 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.900 | 0.500 | 0 | 0.778 | 1.000 | 0.810 | 2,036 |
| 5 | ministral-3:3b | Ollama | native-tools | FR | 0.500 | 1.000 | 0 | 0.611 | N/A* | 0.800 | 7,157 |
| 6 | phi4-mini:3.8b | Ollama | raw-schema | US | 0.700 | 1.000 | 1 | 0.722 | 1.000 | 0.780 | 5,460 |
| 7 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 1 | 0.667 | N/A* | 0.740 | 1,626 |
| **7** | **smollm3:3b** | Ollama | raw-schema | US | 0.600 | 1.000 | 1 | 0.667 | 1.000 | **0.740** | 9,712 |
| 9 | qwen2.5:3b | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.778 | N/A* | 0.670 | 2,801 |
| **9** | **qwen3:1.7b** | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | 0.750 | N/A* | **0.670** | 11,903 |
| **9** | **granite4:3b** | Ollama | native-tools | US | 0.800 | 0.500 | 1 | 0.750 | N/A* | **0.670** | 2,402 |
| 12 | llama3.2:3b | Ollama | native-tools | US | 0.900 | 0.000 | 0 | 0.778 | N/A* | 0.660 | 1,726 |
| 13 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.694 | N/A* | 0.640 | 881 |
| **13** | **functiongemma** | Ollama | native-tools | US | 0.600 | 1.000 | 2 | 0.667 | N/A* | **0.640** | 476 |
| **13** | **lfm2.5:1.2b** | llama.cpp | openai-compat | US | 0.100 | 1.000 | 0 | 0.222 | 0.000 | **0.640** | 1,617 |
| 16 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 | 6,477 |
| 16 | gemma3:1b | Ollama | raw-schema | US | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 | 2,011 |
| 16 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 | 11,362 |
| **19** | **jan-v3:4b** | Ollama | raw-schema | US | 0.100 | 0.500 | 0 | 0.167 | 0.500 | **0.490** | 2,500 |
| **20** | **granite3.3:2b** | Ollama | native-tools | US | 0.700 | 0.000 | 1 | 0.583 | N/A* | **0.480** | 1,650 |
| **21** | **llama3.2:1b** | Ollama | native-tools | US | 0.700 | 0.500 | 3 | 0.667 | N/A* | **0.430** | 1,461 |

\*Ollama native-tools API returns only the first tool call. **Bold rows** are new in Round 2.

### Edge Agent Mini Leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Wrong Tool | Agent Score | Avg ms |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **qwen3:0.6b** | Ollama | native-tools | CN | 0.700 | 1.000 | 0 | **0.880** | 3,645 |
| 2 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 0 | 0.840 | 2,211 |
| 3 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 0.900 | 0.500 | 0 | 0.810 | 2,036 |
| 4 | smollm2:1.7b | Ollama | native-tools | US | 0.600 | 1.000 | 1 | 0.740 | 1,626 |
| **5** | **qwen3:1.7b** | Ollama | native-tools | CN | 0.800 | 0.500 | 1 | **0.670** | 11,903 |
| 6 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.600 | 1.000 | 2 | 0.640 | 881 |
| **6** | **functiongemma** | Ollama | native-tools | US | 0.600 | 1.000 | 2 | **0.640** | 476 |
| **6** | **lfm2.5:1.2b** | llama.cpp | openai-compat | US | 0.100 | 1.000 | 0 | **0.640** | 1,617 |
| 9 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0 | 0.600 | 6,477 |
| 9 | gemma3:1b | Ollama | raw-schema | US | 0.000 | 1.000 | 0 | 0.600 | 2,011 |
| **11** | **llama3.2:1b** | Ollama | native-tools | US | 0.700 | 0.500 | 3 | **0.430** | 1,461 |

### Hard Prompts P10-P12 (detailed)

| Model | P10 Tool | P10 | P11 Tool | P11 | P12 Tool | P12 | Wrong |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b | get_weather | OK | search_files | OK | get_weather | WRONG | 1 |
| qwen2.5:1.5b | (none) | miss | (none) | miss | schedule_meeting | OK | 0 |
| qwen2.5:0.5b | (none) | miss | get_weather | WRONG | get_weather | WRONG | 2 |
| llama3.2:3b | search_files | wrong? | search_files | OK | schedule_meeting | OK | 0 |
| smollm2:1.7b | (none) | miss | (none) | miss | get_weather | WRONG | 1 |
| ministral-3:3b | (none) | miss | (none) | miss | (none) | miss | 0 |
| deepseek-r1:1.5b | (none) | miss | (none) | miss | (none) | miss | 0 |
| gemma3:1b | (none) | miss | (none) | miss | (none) | miss | 0 |
| phi4-mini:3.8b | get_weather | OK | (none) | miss | get_weather | WRONG | 1 |
| bitnet-3B | (none) | miss | (none) | miss | (none) | miss | 0 |
| bitnet-2B-4T | (none) | miss | search_files | OK | schedule_meeting | OK | 0 |
| **qwen3:0.6b** | **(none)** | **miss** | **search_files** | **OK** | **(none)** | **miss** | **0** |
| **qwen3:1.7b** | **get_weather** | **OK** | **search_files** | **OK** | **get_weather** | **WRONG** | **1** |
| **qwen3:4b** | **(none)** | **miss** | **search_files** | **OK** | **(none)** | **miss** | **0** |
| **functiongemma** | **(none)** | **miss** | **get_weather** | **WRONG** | **get_weather** | **WRONG** | **2** |
| **granite3.3:2b** | **get_weather** | **OK** | **(none)** | **miss** | **get_weather** | **WRONG** | **1** |
| **llama3.2:1b** | **schedule_meeting** | **WRONG** | **get_weather** | **WRONG** | **get_weather** | **WRONG** | **3** |
| **lfm2.5:1.2b** | **get_weather** | **OK** | **(none)** | **miss** | **(none)** | **miss** | **0** |
| **granite4:3b** | **get_weather** | **OK** | **search_files** | **OK** | **get_weather** | **WRONG** | **1** |
| **smollm3:3b** | **(none)** | **miss** | **(none)** | **miss** | **get_weather** | **WRONG** | **1** |
| **jan-v3:4b** | **(none)** | **miss** | **(none)** | **miss** | **(none)** | **miss** | **0** |

**Legend:** "OK" = correct tool. "WRONG" = called the specifically-bad tool (penalized). "wrong?" = wrong tool but not the worst choice (not penalized). "miss" = didn't call any tool (no penalty, but no Action credit).

## New Model Analysis

### The Qwen3 Family: Thinking Meets Tool Calling

Qwen3 was the most-requested model family (6 separate users). Three sizes were tested: 0.6B, 1.7B, and 4B. All three use Ollama's native tool API and have built-in thinking capability.

**qwen3:0.6b (0.6B params) -- New Champion, 600 Million Parameters**

The smallest Qwen3 tied for the highest Agent Score in the benchmark (0.880), matching the 4B variant while being 17x faster (3,645 ms vs 63,717 ms average). Perfect restraint on P5 and P9. Zero wrong tool calls. It correctly called `search_files` on P11 (resisting the "weather" keyword trap), showing negation comprehension at a parameter count where most models fail.

Where it falls short: P10 (cycling in Bruges) and P12 (scheduling despite provided weather) were both declined rather than attempted. The model's strategy mirrors qwen2.5:1.5b from Round 1 -- when uncertain, don't act. This conservatism is rewarded by the scoring formula.

On P12, the model showed inconsistency across runs: Run 1 correctly called `schedule_meeting`, but Runs 2 and 3 declined. In one run it even stated "The current tools don't include a function to schedule a meeting" -- contradicting its own tool list. At 600M parameters, working memory for tool schemas appears fragile.

**qwen3:4b (4B params) -- Same Score, 17x Slower**

Tied at 0.880 with identical behavior to the 0.6B on majority-voted results: perfect restraint, zero wrong tools, same P10/P12 misses. The difference is latency. Thinking mode generates extensive reasoning chains that balloon inference time: P7 took 148 seconds, P12 took 162 seconds.

The thinking traces reveal sophisticated reasoning. On P12, the model correctly identified that weather was already provided and that `schedule_meeting` was the right tool, but then concluded it couldn't call the tool because no meeting time was specified:

> "The user's request to schedule a meeting requires a specific time (which is not provided in the query), making it impossible to call the `schedule_meeting` function."

This is technically a valid objection -- the prompt says "Should I schedule an indoor meeting with Jan?" without specifying a time. The model chose not to act rather than hallucinate a time. On P10, similar reasoning: it noted that `get_weather` returns current weather, not forecasts, making it unreliable for a "next Thursday" decision.

Whether this represents superior reasoning or excessive caution depends on the deployment context. The benchmark penalizes inaction (Action 0.700), but in production an agent that refuses impossible tasks may be preferable to one that hallucinates parameters.

**qwen3:1.7b (1.7B params) -- The Middle Child Problem**

The 1.7B scored 0.670, significantly lower than both its siblings. The culprit: P9 restraint failure. When asked to write a Python weather script, it called `get_weather("Antwerp")` in 2 of 3 runs -- keyword-triggered by "weather" in the prompt, the same failure pattern that affected qwen2.5:3b in Round 1.

The model also called `get_weather` on P12 (weather already provided), earning 1 wrong tool penalty. Combined with the restraint failure: 0.800 x 0.4 + 0.500 x 0.3 + 0.667 x 0.3 = 0.670.

This creates a non-monotonic relationship within the Qwen3 family: 0.6B (0.880) > 4B (0.880) > 1.7B (0.670). The 1.7B model appears to sit in a capability valley -- large enough to be aggressive about tool calling, but not large enough to exercise judgment about when not to. The 0.6B model's conservatism and the 4B model's reasoning both avoid the trap that catches the 1.7B.

Latency is also notably high for its size: 11,903 ms average, driven by thinking chains. P7 took 19.5 seconds and P9 took 44.7 seconds.

### functiongemma (270M) -- Purpose-Built, Still Keyword-Trapped

The most anticipated model in Round 2: a 270M fine-tune specifically designed for function calling. Two users predicted it would have "very high performance per compute." At 476 ms average latency, it's the fastest model in the benchmark by a wide margin.

Agent Score: 0.640. It nailed the basics (P1-P3, P7 all correct) and showed perfect restraint on P5 and P9. But on the hard prompts it fell into the same keyword trap as models seven times its size:

- P11: Called `get_weather("Antwerp")` despite being told "don't check the weather." The negation was completely ignored.
- P12: Called `get_weather("Antwerp")` despite the weather being provided in the prompt. Also called `schedule_meeting` in the same response, showing the correct intent was present but secondary to the keyword trigger.

At 270M parameters, functiongemma has the smallest model in the benchmark that can produce valid tool calls. Its restraint is excellent -- it correctly declined P5, P9, and P10 (all three). But it cannot parse negation or detect redundant information. These are the same failures that affect qwen2.5:0.5b (500M), suggesting a capability floor around 500M-1B parameters for contextual tool selection.

### granite4:3b (3B) -- IBM's Quiet Achiever

IBM's latest Granite generation scored 0.670, with a strong Action Score (0.800) and solid hard-prompt performance. It correctly called `get_weather("Bruges")` on P10 (implicit weather reasoning) and `search_files` on P11 (negation comprehension) -- only 5 models in the benchmark pass both.

The single failure: P12, where it called `get_weather("Antwerp")` despite the weather being provided. This is the most common failure across all models.

Its P9 restraint failure (calling `get_weather("San Francisco")` when asked to write a Python weather script) is notable for the consistent hallucinated city across all 3 runs. The model doesn't generate any `raw_content` text for most responses -- it's a pure tool-calling machine that either calls a tool or returns nothing. The one exception is P5 (meta question), where it produced a well-formatted markdown table listing its tools.

The comparison with granite3.3:2b (0.480) is stark. Both are IBM models, but granite4 shows dramatically better judgment: granite3.3 has zero restraint (calls tools on every prompt including P5 and P9), while granite4 passes P5 and shows contextual awareness on P10-P11.

### smollm3:3b (3B) -- Chain-of-Thought Reveals the Gap Between Knowing and Doing

HuggingFace's SmolLM3 scored 0.740, matching smollm2:1.7b. It has perfect restraint (P5, P9 both correct), handles P1-P3 and P6-P8, and achieved Multi-Tool 1.000 on P8.

The model's visible `<think>` blocks are its most distinctive feature. On P5:

> "Okay, the user is asking what tools I have access to... Since this is not a request to perform a task... the appropriate response would be to list them out."

Correct reasoning leading to correct action. But on P10:

> "To determine whether you should take the train or bike for your meeting... we'll need to gather some information..."

The model then writes `{"name": "get_weather", "arguments": {"city": "Bruges"}}` in its text response -- but *without* wrapping it in `<tool_call>` tags, so the parser doesn't capture it. The reasoning identified the correct tool, the JSON was correctly formatted, but the output framing was wrong. This "almost there" pattern recurred on multiple prompts.

On P11 the model claimed "I don't have the capability to search files based on patterns like quarterly reports" -- directly contradicting its own tool list. At 9,712 ms average latency (driven by thinking chains), it's slow for what it delivers.

### lfm2.5:1.2b (1.2B) -- Right Architecture, Wrong Output Format

Liquid AI's state-space hybrid model was recommended by 3 users, with one calling it "a fantastic job for its size." It scored 0.640 with Action 0.100 -- meaning only 1 of 10 actionable prompts produced a parsed tool call.

The model's failure is primarily a format problem, not a comprehension problem. It writes tool calls using bracket notation (`[get_weather(city="Bruges")]`) or inline function-call syntax instead of the `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` format the parser expects. On P10, it correctly reasoned that weather matters for cycling and wrote `[get_weather(city="Bruges")]` -- the right tool, right argument, wrong wrapper. Only when the model happens to produce valid `<tool_call>` tags (1 of 36 responses) does a call register.

On P5 and P9, lfm2.5 showed clean restraint, listing tools in text without calling any. Its meta-understanding of its capabilities is solid. The barrier to deployment is purely one of output formatting -- a custom parser for its bracket syntax could potentially recover most of its intended tool calls, which would significantly improve its Action Score.

### jan-v3:4b (4B) -- Format Failure Across the Board

Jan v3-4B is a Qwen3 fine-tune from jan.ai. It scored 0.490 -- second to last among all models.

The root cause: the model outputs closing `</tool_call>` tags but drops the opening `<tool_call>` tag, causing nearly every tool call to fail parsing. Across 36 responses (12 prompts x 3 runs), only about 6 produced parsed tool calls. The model's *intent* is often correct:

- P11: Outputs `{"name": "search_files", "arguments": {"pattern": "quarterly_report_*.pdf"}}` -- correct tool, reasonable pattern, but missing the opening tag.
- P10: Outputs `{"name": "get_weather", "arguments": {"arg1": "Bruges"}}` -- correct tool, but uses `"arg1"` instead of `"city"` as the parameter name.

The model also requires `think=False` to produce any output at all. With thinking enabled (default), all content goes to the `thinking` field and `content` is empty. This is a fundamental compatibility issue with the raw-prompt backend.

On P5 (meta question), it produced the worst possible response: calling all three tools with fabricated demo data (`get_weather("New York")`, `search_files("*.txt")`, `schedule_meeting("Project Review")`), apparently interpreting "What tools do you have?" as a request to demonstrate each one.

### granite3.3:2b (2B) -- Tool-Calling Machine Without Brakes

IBM's earlier Granite 3.3 scored 0.480, with zero restraint: it calls a tool on every single prompt, including P5 (meta question) and P9 (code-writing request). On P5, Run 2 called *all three tools simultaneously* with fabricated arguments. Almost every response has empty `raw_content` -- the model produces no natural language, just tool calls.

Despite this, its Action Score (0.700) reflects decent tool selection on the easy prompts and P10 (correctly calling `get_weather("Bruges")`). But the zero restraint and a wrong tool call on P12 give it the lowest score among functional models with native tool support.

The contrast with granite4:3b (0.670) shows clear generational improvement. Same company, similar size, but granite4 has restraint on P5, contextual awareness on P11, and natural language responses when appropriate.

### llama3.2:1b (1B) -- Most Chaotic Outputs in the Benchmark

The smallest Llama 3.2 scored 0.430 -- dead last. It has Action 0.700 (calls tools aggressively and often picks the right one for easy prompts), but Wrong Tool 3 (the maximum possible) and partial restraint failure make it unreliable.

Its outputs are the most chaotic in the benchmark:
- P5: Calls `get_weather` with hallucinated cities (London, Berlin, Antwerp -- different each run).
- P9: Outputs raw Python code with leaked `<|python_tag|>` tokens from Llama's training.
- P10: Calls `schedule_meeting` with fabricated attendees (`"client@bruguesurfers.com"`, `"anotherclient@bruguesurfers.com"`).
- P11: Calls `get_weather("")` with an empty city string, plus `search_files("*.csv")` -- searches for CSVs instead of a quarterly report.

Every hard prompt produces the worst possible tool call. On P10 it schedules a meeting that already exists (WRONG). On P11 it checks the weather after being told not to (WRONG). On P12 it re-checks weather already provided (WRONG). At 1B parameters with llama3.2's architecture, the model can produce valid tool-call JSON but has no judgment about what to put in it.

## Score Changes for Original Models

All 11 original models were rerun fresh for Round 2. Some scores shifted significantly due to run-to-run variance in the 3-run majority voting:

| Model | Round 1 | Round 2 | Change | What Changed |
|---|---|---|---|---|
| bitnet-2B-4T | 0.570 | 0.810 | +0.240 | P10: schedule_meeting (WRONG) -> (none); P12: get_weather (WRONG) -> schedule_meeting (OK). Two wrong tools eliminated. |
| smollm2:1.7b | 0.640 | 0.740 | +0.100 | P11: get_weather (WRONG) -> (none). One wrong tool eliminated. |
| phi4-mini:3.8b | 0.680 | 0.780 | +0.100 | P11: get_weather (WRONG) -> (none). One wrong tool eliminated. |
| qwen2.5:1.5b | 0.800 | 0.840 | +0.040 | Minor improvement in action. |
| qwen2.5:3b | 0.670 | 0.670 | 0.000 | Stable. |
| llama3.2:3b | 0.660 | 0.660 | 0.000 | Stable. |
| ministral-3:3b | 0.800 | 0.800 | 0.000 | Stable. |
| deepseek-r1:1.5b | 0.600 | 0.600 | 0.000 | Stable (0/12 tools in both rounds). |
| gemma3:1b | 0.600 | 0.600 | 0.000 | Stable (0/12 tools in both rounds). |
| bitnet-3B | 0.600 | 0.600 | 0.000 | Stable (incoherent in both rounds). |
| qwen2.5:0.5b | 0.640 | 0.640 | 0.000 | Stable. |

The largest shift is bitnet-2B-4T (+0.240). In Round 1, it called `schedule_meeting` on P10 (penalized wrong tool) and `get_weather` on P12 (penalized wrong tool). In the Round 2 rerun, it declined P10 and correctly called `schedule_meeting` on P12. Whether the model calls the wrong tool vs. declines depends on stochastic sampling -- 3 runs is enough to produce stable results on easy prompts but not on the hard prompts where behavior is already borderline.

This variance affects primarily the Wrong Tool metric, which contributes 30% of the Agent Score. Models that are "on the edge" on hard prompts will fluctuate between runs.

## Cross-Model Findings

### P12 Remains the Hardest Prompt

"The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?" -- 3 of 21 models called the correct tool (`schedule_meeting`): qwen2.5:1.5b, llama3.2:3b, and bitnet-2B-4T. Eleven models called `get_weather` (the penalized wrong tool). Seven models declined entirely.

P12 requires three capabilities simultaneously: reading provided context (weather is known), resisting a keyword trigger ("weather"), and identifying the actual requested action (scheduling). No model under 1.5B parameters gets this right.

### The Negation Test (P11) Separates Families

"Don't check the weather in Antwerp, just find me the quarterly report." -- 7 models correctly called `search_files`:

| Model | Size | P11 Result |
|---|---|---|
| qwen3:0.6b | 0.6B | search_files OK |
| qwen3:1.7b | 1.7B | search_files OK |
| qwen3:4b | 4B | search_files OK |
| qwen2.5:3b | 3B | search_files OK |
| granite4:3b | 3B | search_files OK |
| bitnet-2B-4T | 2B | search_files OK |
| llama3.2:3b | 3B | search_files OK |

All three Qwen3 sizes pass P11, as do both larger Qwen2.5 and Granite4. The models that fail P11 by calling `get_weather` (qwen2.5:0.5b, functiongemma, llama3.2:1b, granite3.3:2b) all have either very small parameter counts or were designed without negation training.

### Thinking Mode: A Double-Edged Sword

Four models in Round 2 have thinking capability: qwen3:0.6b, qwen3:1.7b, qwen3:4b, and smollm3:3b. Their thinking adds latency but doesn't consistently improve judgment:

| Model | Agent Score | Avg Latency | Latency/Score Ratio |
|---|---|---|---|
| qwen3:0.6b | 0.880 | 3,645 ms | 4,142 ms/point |
| qwen3:4b | 0.880 | 63,717 ms | 72,406 ms/point |
| smollm3:3b | 0.740 | 9,712 ms | 13,124 ms/point |
| qwen3:1.7b | 0.670 | 11,903 ms | 17,766 ms/point |

qwen3:0.6b achieves the best latency/score ratio of any thinking model. qwen3:4b spends 17x more time thinking for the same score. smollm3:3b's thinking traces show correct reasoning that fails in execution -- it identifies the right tool but doesn't wrap the call in proper tags.

For tool calling specifically, longer thinking chains don't appear to help. The decisions are fast pattern matches (which tool fits this prompt?), not multi-step reasoning problems. The thinking overhead is mostly wasted on prompts where the answer is obvious, and doesn't rescue the model on prompts where the answer requires contextual understanding that the model lacks.

### Format Failures Are a Major Category

Three models in Round 2 lost most of their tool calls to formatting issues rather than comprehension failures:

| Model | Intent Correct | Parsed Correctly | Format Issue |
|---|---|---|---|
| lfm2.5:1.2b | ~6/12 prompts | 1/12 prompts | Bracket notation `[tool(args)]` instead of XML tags |
| jan-v3:4b | ~5/12 prompts | ~2/12 prompts | Missing opening `<tool_call>` tag |
| smollm3:3b | ~8/12 prompts | 6/12 prompts | Tool call JSON in text body without tags |

These models understand what tool to call but can't produce the output format the parser expects. A more permissive parser (or model-specific output adapters) could significantly improve their scores. This suggests that raw tool-calling evaluation is partly measuring format compliance rather than reasoning capability.

### Speed vs. Judgment Frontier

| Model | Agent Score | Avg ms | Param |
|---|---|---|---|
| functiongemma | 0.640 | 476 | 270M |
| qwen2.5:0.5b | 0.640 | 881 | 500M |
| qwen2.5:1.5b | 0.840 | 2,211 | 1.5B |
| qwen3:0.6b | 0.880 | 3,645 | 600M |
| bitnet-2B-4T | 0.810 | 2,036 | 2B |
| granite4:3b | 0.670 | 2,402 | 3B |

functiongemma is the fastest model by far (476 ms) but its judgment on hard prompts limits it to 0.640. qwen3:0.6b achieves the best score (0.880) at moderate latency (3,645 ms). The Pareto frontier for "fastest at each score level" runs through functiongemma (0.640/476ms), qwen2.5:1.5b (0.840/2,211ms), and qwen3:0.6b (0.880/3,645ms).

For latency-critical deployments under 1 second, functiongemma and qwen2.5:0.5b are the only options, both at 0.640.

## Conclusions

1. **A 600M parameter model leads the benchmark.** qwen3:0.6b ties for #1 at 0.880 with its 4B sibling while running 17x faster. Its success comes from the same strategy that won Round 1 for qwen2.5:1.5b -- conservative declining on uncertain prompts -- but with better tool selection on the prompts it does attempt. The Qwen3 architecture and training, even at 600M, produces better judgment than most 3B+ models in the benchmark.

2. **Parameter count is a weak predictor of tool-calling quality.** Rankings within the Qwen3 family are non-monotonic: 0.6B (0.880) > 4B (0.880) > 1.7B (0.670). Across all models, the correlation between parameter count and Agent Score is weak. functiongemma (270M) ties with qwen2.5:0.5b (500M) and lfm2.5:1.2b (1.2B). llama3.2:1b (1B) scores lower than qwen3:0.6b (600M). Architecture and training data composition appear to matter more than raw size for tool-calling judgment in the sub-4B range.

3. **Purpose-built doesn't mean best.** functiongemma was fine-tuned specifically for function calling. It achieved the fastest latency (476 ms) and perfect restraint, but fell into the same keyword traps as generic models on the hard prompts (Wrong Tool 2). Fine-tuning for tool-call format compliance doesn't appear to help with contextual judgment about *which* tool to call.

4. **Generational improvement is real.** granite3.3:2b (0.480) vs granite4:3b (0.670); smollm2:1.7b (0.740) vs smollm3:3b (0.740, matching score but with Multi-Tool 1.000). Both IBM and HuggingFace show clear improvements between model generations on the same task.

5. **Format compliance is a separate axis from reasoning capability.** Three models (lfm2.5, jan-v3, smollm3) lost significant credit because they can reason about tools correctly but can't produce the expected output format. This is a real deployment concern -- the output format contract matters as much as the model's internal reasoning -- but it means scores here partially reflect format training rather than pure agent capability.

6. **3-run majority voting has high variance on edge cases.** bitnet-2B-4T shifted from 0.570 to 0.810 between Round 1 and Round 2 reruns, entirely due to different outcomes on P10 and P12. The hard prompts are where this variance concentrates, because models that are borderline on a prompt will flip between calling the right tool, the wrong tool, or no tool depending on sampling. More runs would stabilize these scores, at the cost of longer benchmark time.

7. **The conservative strategy still wins under this scoring formula.** The top 3 models (qwen3:0.6b, qwen3:4b, qwen2.5:1.5b) all have the same pattern: perfect restraint, zero wrong tools, moderate Action. The formula gives 60% combined weight to restraint and wrong-tool-avoidance, structurally favoring models that decline uncertain prompts. Under an action-maximizing formula (e.g., Action x 0.7 + Restraint x 0.15 + WTA x 0.15), aggressive models like bitnet-2B-4T (Action 0.900) and llama3.2:3b (Action 0.900) would rank higher. The "right" formula depends on the deployment context: autonomous agents should be conservative; human-in-the-loop agents can be aggressive.

8. **The community-requested models mostly confirmed the original findings rather than overturning them.** The Qwen family dominance extended from Qwen2.5 to Qwen3. The keyword-trap failure pattern on P11/P12 appeared in the new models at similar rates. No new model broke the P12 barrier (only 3 of 21 get it right). The most surprising result is qwen3:0.6b showing that sub-1B models can lead a tool-calling benchmark -- but only if they know when not to act.
