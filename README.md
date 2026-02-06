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

- **Tool calls:** Did the model call a tool? Which one?
- **Valid args:** Were the arguments well-formed JSON with the right fields?
- **Latency:** Wall-clock time per inference call (milliseconds).
- **Restraint:** On prompts where the correct answer is "don't call a tool," did the model correctly abstain? Scored as X/2 (P5 and P9).
- **Agent Score:** A composite metric: `(valid_tool_calls / 7) * 0.5 + (restraint / 2) * 0.5`. Weights accuracy and restraint equally. A model that calls tools on everything maxes out at 0.500. You need both to score well.

Everything is run 3 times with majority-vote aggregation to smooth out non-determinism.

## Results at a glance

| Rank | Model | Origin | Tool Calls | Valid Args | Avg Latency | Restraint | Agent Score |
|---|---|---|---|---|---|---|---|
| 1 | qwen2.5:3b | CN | 6/9 | 6/6 | 3,861 ms | 2/2 | **0.929** |
| 1 | qwen2.5:0.5b | CN | 6/9 | 6/6 | 1,351 ms | 2/2 | **0.929** |
| 1 | smollm2:1.7b | US | 6/9 | 6/6 | 2,437 ms | 2/2 | **0.929** |
| 1 | phi4-mini:3.8b | US | 6/9 | 6/6 | 5,723 ms | 2/2 | **0.929** |
| 5 | ministral-3:3b | FR | 5/9 | 5/5 | 10,571 ms | 2/2 | **0.857** |
| 6 | qwen2.5:1.5b | CN | 4/9 | 4/4 | 3,126 ms | 2/2 | 0.786 |
| 7 | bitnet-2B-4T | US | 8/9 | 8/8 | 2,806 ms | 1/2 | 0.750 |
| 8 | llama3.2:3b | US | 9/9 | 9/9 | 2,786 ms | 0/2 | 0.500 |
| 8 | deepseek-r1:1.5b | CN | 0/9 | 0/0 | 7,535 ms | 2/2 | 0.500 |
| 8 | gemma3:1b | US | 0/9 | 0/0 | 4,139 ms | 2/2 | 0.500 |
| 8 | bitnet-3B | US | 0/9 | 0/0 | 16,046 ms | 2/2 | 0.500 |

### Edge agent mini leaderboard (sub-2B models)

| Rank | Model | Avg Latency | Restraint | Agent Score |
|---|---|---|---|---|
| 1 | qwen2.5:0.5b | 1,351 ms | 2/2 | **0.929** |
| 2 | smollm2:1.7b | 2,437 ms | 2/2 | **0.929** |
| 3 | qwen2.5:1.5b | 3,126 ms | 2/2 | 0.786 |
| 4 | bitnet-2B-4T | 2,806 ms | 1/2 | 0.750 |
| 5 | deepseek-r1:1.5b | 7,535 ms | 2/2 | 0.500 |
| 6 | gemma3:1b | 4,139 ms | 2/2 | 0.500 |

## What we learned

### 1. A 400MB model ties a 3.8B model on agent quality

Qwen 0.5B and Phi4-mini both score 0.929. One runs in 1.35 seconds, the other in 5.7. The small model isn't "good for its size" -- it's just good. Same tool-call accuracy, same restraint, 4x faster. SmolLM2 at 1.7B also ties them. Four models from three different organisations (Alibaba, HuggingFace, Microsoft) all converged on the same score using different architectures and training approaches. The ceiling on this benchmark isn't parameter count -- it's the prompts.

### 2. Instruction-tuned BitNet does tool calling

This is the headline result. BitNet-2B-4T went from literal word salad (the base model produces things like "8.- the: ( with a eight the a to as") to 8/8 valid JSON arguments across 9 prompts. A 1.58-bit model -- every weight constrained to {-1, 0, 1}, no floating-point multiplication -- producing perfectly structured `{"name": "get_weather", "arguments": {"city": "Antwerp"}}` on CPU at 2.8 seconds average. As far as we can tell, nobody has benchmarked structured tool-calling on 1-bit models before.

The restraint problem (1/2 -- it called a hallucinated tool on the meta-question P5) is solvable with fine-tuning. The core capability -- understanding a tool schema from a system prompt, selecting the right tool, extracting parameters from natural language, serialising them as valid JSON -- is definitively there at 1.58 bits.

### 3. The 1-bit model understood the hardest prompt better than anyone

P8 asks for two tools at once: "Search for all files matching '*.py' and also tell me the weather in Paris." Every Ollama model -- Qwen, LLaMA, SmolLM, Ministral, Phi4 -- returned a single tool call. That's not a model limitation, it's an API one: Ollama's native tool-calling interface returns only the first tool call, and the raw prompt models only had the first `<tool_call>` tag parsed.

BitNet 2B-4T was the only model that emitted two tool calls back-to-back:
```
<tool_call>{"name": "search_files", "arguments": {"pattern": "*.py"}}
<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}
```

It did this consistently across all 3 runs. That's not a parsing detail -- it means the 1.58-bit model understood the dual intent of the prompt better than models 6x its effective size. The other models have this capability locked behind API constraints; BitNet, running through raw text generation, wasn't limited by them.

### Other findings

**LLaMA 3.2 calls tools on everything.** 9/9 tool calls, 0/2 restraint. It called `search_files` when asked "What tools do you have access to?" and again when asked to write a Python script. It's a hammer that sees every prompt as a nail. Perfect accuracy, zero judgment. Agent Score: 0.500.

**The right backend matters as much as the model.** Phi4-mini scored 0.571 using Ollama's native tool API (it only called 1 tool in 9 prompts). Switching to the raw prompt backend -- same model, same prompts, just a system prompt instead of the `tools=` parameter -- brought it to 0.929. Three models (deepseek-r1, gemma3, phi4-mini) had to be moved off the native API entirely because they either errored or refused to use it.

**Chain-of-thought doesn't help tool calling at small scale.** DeepSeek-R1:1.5B thinks hard (7.5s average latency) and produces responses like `get_weather(Antwerp)` -- it understands the concept but can't produce the JSON format. The reasoning overhead crowds out format compliance at 1.5B params.

**Gemma3:1B gets tantalizingly close.** It outputs `<tool_call>get_weather(city: Antwerp)</tool_call>` -- right tags, right tool, right argument, wrong serialisation (Python kwargs instead of JSON). A custom parser for its format could potentially recover these calls.

**Ministral-3:3B is accurate but slow.** 5/5 valid args, perfect restraint, but 10.6s average latency with some prompts taking 30+ seconds. The EU sovereignty candidate works, but you'll wait for it.

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
