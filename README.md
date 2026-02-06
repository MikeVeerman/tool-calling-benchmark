# Local Agent Bench

**Can a $1,000 laptop run an AI agent that knows when to use tools -- and when not to?**

This benchmark answers that question. No cloud API. No GPU. Just Ollama, a handful of 1-bit and 4-bit quantised models, and a Framework 13 running Arch Linux.

## Why this exists

Tool-calling is the backbone of AI agents. An LLM that can reliably decide "this prompt needs `get_weather`, that one needs `schedule_meeting`, and this other one needs *nothing at all*" is the difference between a useful agent and an expensive autocomplete.

Cloud models handle this well. But what about local models running on your laptop's CPU? The small open-weight models (0.5B-3.8B parameters) that Ollama makes trivially easy to run -- can they actually *do* this?

And what about the new wave of 1-bit models? Microsoft's BitNet promises dramatic efficiency gains through 1.58-bit quantisation. Does that efficiency translate to capability, or is it just fast nonsense?

This benchmark tests all of that: 11 models from 7 organisations across 4 countries, 9 prompts, 3 runs each, on a machine with no discrete GPU.

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

The benchmark uses 9 prompts that escalate in difficulty:

**Easy (P1-P3):** Direct tool calls. "What's the weather in Antwerp?" should obviously call `get_weather`. These establish whether a model can do the basics.

**Ambiguous (P4):** "I'm heading to Brussels tomorrow, anything I should know?" -- calling `get_weather` is reasonable but not required. This tests whether models make sensible judgment calls.

**Restraint (P5, P9):** Prompts where the *correct* answer is to NOT call a tool. P5 asks "What tools do you have access to?" (a meta question). P9 asks "Can you write a Python script that checks the weather using an API?" (a code-writing request that mentions "weather" as a keyword trap). These are the most interesting tests -- an agent that calls tools when it shouldn't is worse than one that occasionally misses a valid call.

**Hard (P6-P8):** P6 requires context the model doesn't have ("the city where we have our next sprint review"). P7 buries meeting parameters in messy natural language with filler words. P8 asks for two tools at once ("search files AND tell me the weather") to see if models handle multi-tool requests or just pick one.

## What we measure

- **Action Score:** correct_tool_calls / 7. How many of the 7 actionable prompts produced valid tool calls. Measures execution capability.
- **Restraint Score:** correct_refusals / 2. How many of the 2 restraint prompts (P5, P9) were correctly left without a tool call. Measures policy calibration.
- **Reliability:** Average per-prompt (successful_runs / 3). Computed from per-run data *before* majority voting. A model that passes a prompt in 2 of 3 runs gets 0.67 reliability for that prompt, even though majority voting calls it a pass. Measures deployability.
- **Multi-Tool Accuracy:** correct_tools / required_tools for P8 (dual-tool prompt). P8 requires both `search_files` and `get_weather`. Ollama's native tool API returns only the first tool call, so this is N/A for native-tools models.
- **Agent Score:** A derived composite: `Action * 0.5 + Restraint * 0.5`. Retained for backward compatibility. A model that calls tools on everything maxes out at 0.500. You need both action and restraint to score well.
- **Latency:** Wall-clock time per inference call (milliseconds).

Everything is run 3 times. Correctness uses majority-vote aggregation; reliability uses per-run data.

## Results at a glance

| Rank | Model | Backend | Mode | Origin | Action | Restraint | Reliability | Multi-Tool | Agent Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:3b | Ollama | native-tools | CN | 0.857 | 1.000 | 0.815 | N/A* | **0.929** |
| 1 | qwen2.5:0.5b | Ollama | native-tools | CN | 0.857 | 1.000 | 0.852 | N/A* | **0.929** |
| 1 | smollm2:1.7b | Ollama | native-tools | US | 0.857 | 1.000 | 0.889 | N/A* | **0.929** |
| 1 | phi4-mini:3.8b | Ollama | raw-schema | US | 0.857 | 1.000 | 0.926 | N/A&dagger; | **0.929** |
| 5 | ministral-3:3b | Ollama | native-tools | FR | 0.714 | 1.000 | 0.778 | N/A* | **0.857** |
| 6 | qwen2.5:1.5b | Ollama | native-tools | CN | 0.571 | 1.000 | 0.704 | N/A* | 0.786 |
| 7 | bitnet-2B-4T | bitnet.cpp | openai-compat | US/1bit | 1.000 | 0.500 | 0.926 | 1.000 | 0.750 |
| 8 | llama3.2:3b | Ollama | native-tools | US | 1.000 | 0.000 | 0.778 | N/A* | 0.500 |
| 8 | deepseek-r1:1.5b | Ollama | raw-schema | CN | 0.000 | 1.000 | 0.222 | N/A&dagger; | 0.500 |
| 8 | gemma3:1b | Ollama | raw-schema | US | 0.000 | 1.000 | 0.259 | N/A&dagger; | 0.500 |
| 8 | bitnet-3B | bitnet.cpp | openai-compat | US/1bit | 0.000 | 1.000 | 0.222 | N/A&dagger; | 0.500 |

\*Ollama native-tools API returns only the first tool call. &dagger;Raw output not preserved from original run.

### Edge agent mini leaderboard (sub-2B models)

| Rank | Model | Backend | Mode | Action | Restraint | Reliability | Multi-Tool | Agent Score |
|---|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:0.5b | Ollama | native-tools | 0.857 | 1.000 | 0.852 | N/A* | **0.929** |
| 2 | smollm2:1.7b | Ollama | native-tools | 0.857 | 1.000 | 0.889 | N/A* | **0.929** |
| 3 | qwen2.5:1.5b | Ollama | native-tools | 0.571 | 1.000 | 0.704 | N/A* | 0.786 |
| 4 | bitnet-2B-4T | bitnet.cpp | openai-compat | 1.000 | 0.500 | 0.926 | 1.000 | 0.750 |
| 5 | deepseek-r1:1.5b | Ollama | raw-schema | 0.000 | 1.000 | 0.222 | N/A&dagger; | 0.500 |
| 6 | gemma3:1b | Ollama | raw-schema | 0.000 | 1.000 | 0.259 | N/A&dagger; | 0.500 |

## What we learned

### 1. Reliability separates the 0.929 club

Four models share a 0.929 Agent Score, but reliability tells a different story. phi4-mini (0.926) is rock-solid -- only one failure across 27 calls. smollm2 (0.889) is close behind. qwen2.5:0.5b (0.852) drops further. qwen2.5:3b (0.815) is the least reliable of the group -- its P9 restraint failed in 2 of 3 runs, masked by majority voting. Same score, meaningfully different deployment risk.

### 2. A 400MB model ties a 3.8B model on agent quality

Qwen 0.5B and Phi4-mini both score 0.929. One runs in 1.35 seconds, the other in 5.7. The small model isn't "good for its size" -- it's just good. Same Action Score, same Restraint Score, 4x faster. SmolLM2 at 1.7B also ties them. Four models from three different organisations (Alibaba, HuggingFace, Microsoft) all converged on the same score using different architectures and training approaches. The ceiling on this benchmark isn't parameter count -- it's the prompts.

### 3. Instruction-tuned BitNet is a strong actuator with weak policy calibration

BitNet-2B-4T: Action 1.000 (perfect execution), Restraint 0.500 (P5 fails), Reliability 0.926, Multi-Tool 1.000 (the only model to correctly emit both tools on P8). A 1.58-bit model -- every weight constrained to {-1, 0, 1}, no floating-point multiplication -- producing perfectly structured JSON on CPU at 2.8 seconds average. Its weakness is policy calibration (it doesn't know when *not* to call a tool), not execution capability. As far as we can tell, nobody has benchmarked structured tool-calling on 1-bit models before.

### 4. The 1-bit model understood the hardest prompt better than anyone

P8 asks for two tools at once. Every Ollama model returned a single tool call -- some due to API constraints (native-tools only returns the first call), some due to parser limitations. BitNet 2B-4T emitted both calls back-to-back, consistently across all 3 runs, achieving Multi-Tool Accuracy of 1.000. The other models' Multi-Tool scores are N/A because the native API didn't capture the second call.

### 5. This benchmark tests model-protocol pairs, not models in isolation

phi4-mini went from 0.571 to 0.929 by switching from Ollama's native tool API to a raw system prompt. The Backend and Mode columns in the leaderboard exist to make this dependency explicit. Comparing models without noting their interaction contract is misleading.

### Other findings

**LLaMA 3.2 calls tools on everything.** Action 1.000, Restraint 0.000, Reliability 0.778. It called `search_files` when asked "What tools do you have access to?" and again when asked to write a Python script. Perfect execution, zero judgment. Agent Score: 0.500.

**Chain-of-thought doesn't help tool calling at small scale.** DeepSeek-R1:1.5B thinks hard (7.5s average latency) and produces responses like `get_weather(Antwerp)` -- it understands the concept but can't produce the JSON format. The reasoning overhead crowds out format compliance at 1.5B params.

**Gemma3:1B gets tantalizingly close.** It outputs `<tool_call>get_weather(city: Antwerp)</tool_call>` -- right tags, right tool, right argument, wrong serialisation (Python kwargs instead of JSON). A custom parser for its format could potentially recover these calls.

**Ministral-3:3B is accurate but slow.** Action 0.714, Restraint 1.000, but 10.6s average latency with some prompts taking 30+ seconds. The EU sovereignty candidate works, but you'll wait for it.

**P9 remains the best discriminator prompt.** "Can you write a Python script that checks the weather using an API?" cleanly separates models that understand user intent from models that pattern-match on keywords. Only LLaMA 3.2 failed it consistently.

For the full data -- per-run breakdowns, latency matrices, raw BitNet output samples, failure analysis -- see [REPORT.md](REPORT.md).

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

The full run (11 models x 9 prompts x 3 runs = 297 inference calls) takes roughly 45-60 minutes on the hardware described above. Most of that time is BitNet 3B (base model) and ministral-3:3b -- the other models finish faster.

### Customising

The benchmark is a single file (`bench.py`). To add models, prompts, or adjust runs:

- **Add an Ollama model (native tool API):** Add `{"name": "your-model:tag", "backend": "ollama", "origin": "XX"}` to `ALL_MODELS`
- **Add an Ollama model (raw prompt):** Use `"backend": "ollama_raw"` for models that don't support Ollama's native tool API or perform better with system-prompt-based tool calling
- **Add a prompt:** Append to `TEST_PROMPTS`. If it's a restraint prompt (correct answer is no tool call), add its 0-based index to `RESTRAINT_INDICES`
- **Change run count:** Edit `num_runs` in `main()`
- **Add tools:** Extend the `TOOLS` list and `TOOL_DISPATCH` dict. Update `BITNET_SYSTEM_PROMPT` if you want raw-prompt models to know about them
- **Add to edge leaderboard:** Add the model name to `EDGE_MODELS` if it's sub-2B

## License

Do whatever you want with this. It's a benchmark script, not a product.
