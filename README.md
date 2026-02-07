# Local Agent Bench

**Can a $1,000 laptop run an AI agent that knows when to use tools -- and when not to?**

This benchmark answers that question. No cloud API. No GPU. Just Ollama, a handful of 1-bit and 4-bit quantised models, and a Framework 13 running Arch Linux.

## Why this exists

Tool-calling is the backbone of AI agents. An LLM that can reliably decide "this prompt needs `get_weather`, that one needs `schedule_meeting`, and this other one needs *nothing at all*" is the difference between a useful agent and an expensive autocomplete.

But there's a harder question: when a prompt mentions "weather" but the correct action is *not* to call `get_weather`, can the model resist the keyword trigger? When the user says "don't check the weather, just find the report," does the model listen? When the weather is already provided in the prompt, does the model notice?

Cloud models handle this well. But what about local models running on your laptop's CPU? The small open-weight models (0.5B-3.8B parameters) that Ollama makes trivially easy to run -- can they actually *do* this?

This benchmark tests all of that: 11 models from 7 organisations across 4 countries, 12 prompts, 3 runs each, on a machine with no discrete GPU.

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

### Qwen 2.5 (3B, 1.5B, 0.5B) -- the scaling ladder

Qwen 2.5 is one of the strongest open model families for tool-calling at small sizes. Alibaba specifically trained these for instruction-following and function-calling. By testing all three sizes we get a clean read on how much capability you lose as you shrink from 3B down to 0.5B parameters.

### LLaMA 3.2:3B -- Meta's contender

Meta's LLaMA 3.2 at 3B parameters is the obvious comparison point. It has native tool-calling support in Ollama and is widely used. It's the model most people would reach for first.

### SmolLM2:1.7B -- the underdog

HuggingFace's SmolLM2 is purpose-built to be small and fast. At 1.7B parameters it sits between Qwen's 1.5B and 3B. A newer model that doesn't get as much benchmark attention. Including it tests whether the "small model" space has dark horses.

### Ministral-3:3B -- the EU sovereignty candidate

Mistral's 3B edge model, Apache 2.0 licensed, from the French AI lab. This is the model you'd pick if you needed tool-calling with a European-sourced model. Tests whether the non-US/CN model ecosystem has caught up.

### DeepSeek-R1:1.5B -- the reasoning distillation

DeepSeek's distilled reasoning model uses chain-of-thought internally. The interesting question: does thinking before answering improve restraint or just burn tokens?

### Gemma3:1B -- Google's smallest

Google's smallest instruction model with a different architecture (sliding window attention). At 1B parameters, it tests the floor for tool-calling capability.

### Phi4-mini:3.8B -- Microsoft's reasoning model

Slightly larger than the 3B tier but trained specifically for structured reasoning. Tests whether Microsoft's training approach translates to tool-calling.

### BitNet b1.58-3B -- the 1-bit base model

Microsoft's BitNet uses 1.58-bit quantisation ({-1, 0, 1} ternary weights). The publicly available `bitnet_b1_58-3B` is a **base model**, not instruction-tuned. Included as a control to see what raw 1-bit inference looks like before instruction tuning.

### BitNet b1.58-2B-4T -- the 1-bit instruction-tuned model

The same 1.58-bit architecture, but instruction-tuned on 4 trillion tokens. This is the model that answers the question everyone has been asking: can ternary weights produce structured output?

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
- **Reliability:** Average per-prompt (successful_runs / 3). Computed from per-run data *before* majority voting. A model that passes a prompt in 2 of 3 runs gets 0.67 reliability for that prompt, even though majority voting calls it a pass. Measures deployability.
- **Multi-Tool Accuracy:** correct_tools / required_tools for P8 (dual-tool prompt). P8 requires both `search_files` and `get_weather`. Ollama's native tool API returns only the first tool call, so this is N/A for native-tools models.
- **Agent Score:** `Action × 0.4 + Restraint × 0.3 + Wrong-Tool-Avoidance × 0.3` where Wrong-Tool-Avoidance = (3 - wrong_tool_count) / 3. A model that calls tools aggressively but picks the wrong ones is penalized. A model that conservatively declines uncertain prompts is rewarded.
- **Latency:** Wall-clock time per inference call (milliseconds).

Everything is run 3 times. Correctness uses majority-vote aggregation; reliability uses per-run data.

## Results at a glance

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

### Edge agent mini leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Action | Restraint | Wrong Tool | Reliability | Multi-Tool | Agent Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:1.5b | Ollama | native-tools | 0.500 | 1.000 | 0 | 0.611 | N/A* | **0.800** |
| 2 | qwen2.5:0.5b | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.667 | N/A* | 0.640 |
| 2 | smollm2:1.7b | Ollama | native-tools | 0.600 | 1.000 | 2 | 0.694 | N/A* | 0.640 |
| 4 | deepseek-r1:1.5b | Ollama | raw-schema | 0.000 | 1.000 | 0 | 0.167 | 0.000 | 0.600 |
| 4 | gemma3:1b | Ollama | raw-schema | 0.000 | 1.000 | 0 | 0.194 | 0.000 | 0.600 |
| 6 | bitnet-2B-4T | bitnet.cpp | openai-compat | 0.800 | 0.500 | 2 | 0.750 | 1.000 | 0.570 |

## What we learned

### 1. Hard prompts broke the plateau

In Round 3, four models tied at 0.929 Agent Score. P10-P12 spread them from 0.640 to 0.800, with two new leaders -- qwen2.5:1.5b and ministral-3:3b -- that weren't even in the top group before. The Round 3 ceiling was an artifact of the benchmark not being hard enough. Adding three prompts that test judgment rather than execution completely reshuffled the leaderboard.

### 2. Not calling a tool is better than calling the wrong one

The two leaders scored highest by *declining* uncertain prompts rather than guessing wrong. qwen2.5:1.5b missed P10 and P11 entirely (losing Action points) but avoided wrong-tool penalties. ministral-3:3b did the same. Meanwhile, models that aggressively called tools -- qwen2.5:0.5b, smollm2, phi4-mini -- all got 2 wrong tool calls, dropping them below the conservative models. In the real world, an agent that does nothing when confused is safer than one that takes the wrong action.

### 3. Keyword matching appears to be a common failure pattern

Five of eight functional models called `get_weather` whenever they saw "weather" in the prompt, regardless of context. P11 says "don't check the weather" -- three models called `get_weather` anyway. P12 says "the weather is 8°C and rainy" (already known) -- five models called `get_weather` to look it up again. This is consistent with sub-4B models relying on keyword matching rather than semantic understanding for tool selection, though the sample (3 hard prompts, 3 runs) is small.

### 4. Bigger isn't always better

qwen2.5:1.5b (0.800) now outperforms qwen2.5:3b (0.670). The 3B model's aggression -- calling `get_weather` when asked to write code (P9) and when weather was already given (P12) -- costs more than the 1.5B model's conservatism. The relationship between parameter count and agent quality is non-monotonic when judgment is measured. Note: this ranking depends on the scoring formula, which gives 60% combined weight to restraint and wrong-tool-avoidance. Under an action-heavy formula, the 3B model would rank higher. The underlying observation is robust: the larger model is more aggressive and makes more wrong calls; the smaller model is more conservative and avoids them.

### 5. The 1-bit model is a strong executor with weak judgment

BitNet-2B-4T retained its execution prowess: Action 0.800, Multi-Tool 1.000 (still the only model to correctly emit both tools on P8). But the judgment prompts were devastating: it called `schedule_meeting` for P10 (should be `get_weather`) and `get_weather` for P12 (weather already given). A 1.58-bit model can generate perfectly structured JSON on CPU at 2.3s average, but this 2B model's judgment on P10-P12 is weak — whether that's due to the ternary weights, the parameter count, or the training data can't be isolated from this benchmark.

### Other findings

**LLaMA 3.2 has better judgment than expected.** Despite zero restraint (calls a tool on every prompt), it correctly picked `search_files` for P11 and `schedule_meeting` for P12 -- the two prompts that tripped up most models. Action 0.900 is the highest in the benchmark. Its problem is restraint (0.000), not tool selection.

**P12 is the best discriminator.** "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?" Only 2 of 11 models correctly called `schedule_meeting`. It simultaneously tests context awareness, keyword resistance, and action identification.

**P10 is the hardest prompt.** Only 1 model (qwen2.5:3b) called the correct tool. "Should I take the train or cycle?" requires inferring that transport depends on weather -- a reasoning chain no other model made.

**Chain-of-thought still doesn't help.** DeepSeek-R1:1.5B thinks hard (5.8s average latency) and produces responses like `get_weather(Antwerp)` -- it understands the concept but can't produce the JSON format. Same story as Round 3.

**Ministral-3:3B is accurate but slow.** Action 0.500, Restraint 1.000, Wrong Tool 0 -- but 8.3s average latency with some prompts taking 29+ seconds. The EU sovereignty candidate works safely, but you'll wait for it.

For the full data -- per-run breakdowns, latency matrices, raw BitNet output samples, hard prompt analysis, failure analysis -- see [REPORT.md](REPORT.md).

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

# Update BITNET_DIR in bench.py if your BitNet path differs
python bench.py
```

The full run (11 models x 12 prompts x 3 runs = 396 inference calls) takes roughly 45-60 minutes on the hardware described above. Most of that time is BitNet 3B (base model) and ministral-3:3b -- the other models finish faster.

### Customising

The benchmark is a single file (`bench.py`). To add models, prompts, or adjust runs:

- **Add an Ollama model (native tool API):** Add `{"name": "your-model:tag", "backend": "ollama", "origin": "XX"}` to `ALL_MODELS`
- **Add an Ollama model (raw prompt):** Use `"backend": "ollama_raw"` for models that don't support Ollama's native tool API or perform better with system-prompt-based tool calling
- **Add a prompt:** Append to `TEST_PROMPTS`. If it's a restraint prompt (correct answer is no tool call), add its 0-based index to `RESTRAINT_INDICES`. If it's a hard prompt with an expected tool, add it to `EXPECTED_TOOLS` and `WRONG_TOOL_MAP`
- **Change run count:** Edit `num_runs` in `main()`
- **Add tools:** Extend the `TOOLS` list and `TOOL_DISPATCH` dict. Update `BITNET_SYSTEM_PROMPT` if you want raw-prompt models to know about them
- **Add to edge leaderboard:** Add the model name to `EDGE_MODELS` if it's sub-2B

## License

Do whatever you want with this. It's a benchmark script, not a product.
