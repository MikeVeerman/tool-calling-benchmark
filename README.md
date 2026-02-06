# Local Agent Bench

**Can a $1,000 laptop run an AI agent that knows when to use tools -- and when not to?**

This benchmark answers that question. No cloud API. No GPU. Just Ollama, a 1.58-bit quantised model, and a Framework 13 running Arch Linux.

## Why this exists

Tool-calling is the backbone of AI agents. An LLM that can reliably decide "this prompt needs `get_weather`, that one needs `schedule_meeting`, and this other one needs *nothing at all*" is the difference between a useful agent and an expensive autocomplete.

Cloud models handle this well. But what about local models running on your laptop's CPU? The small open-weight models (0.5B-3B parameters) that Ollama makes trivially easy to run -- can they actually *do* this?

And what about the new wave of 1-bit models? Microsoft's BitNet promises dramatic efficiency gains through 1.58-bit quantisation. Does that efficiency translate to capability, or is it just fast nonsense?

This benchmark tests all of that: 6 models, 9 prompts, 3 runs each, on a machine with no discrete GPU.

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

**The hypothesis:** The 3B will be most accurate, the 0.5B will be fastest, and the 1.5B will be the sweet spot.

### LLaMA 3.2:3B -- Meta's contender

Meta's LLaMA 3.2 at 3B parameters is the obvious comparison point. It has native tool-calling support in Ollama and is widely used. It's the model most people would reach for first.

**The hypothesis:** Solid tool-calling, competitive with Qwen 3B, possibly better latency.

### SmolLM2:1.7B -- the underdog

HuggingFace's SmolLM2 is purpose-built to be small and fast. At 1.7B parameters it sits between Qwen's 1.5B and 3B. It's a newer model that doesn't get as much attention in benchmarks. Including it tests whether the "small model" space has dark horses.

**The hypothesis:** Functional but probably behind the Qwen models on accuracy.

### BitNet b1.58-3B -- the 1-bit wildcard

Microsoft's BitNet uses 1.58-bit quantisation ({-1, 0, 1} ternary weights), which in theory lets a 3B model run in under 1GB of memory. It's a fundamentally different approach to model compression.

There's a catch: the publicly available `bitnet_b1_58-3B` is a **base model**, not instruction-tuned. It has never been trained to follow prompts, understand tool schemas, or produce structured output. We integrate it via a `llama-server` subprocess with a system prompt that describes the tools in text and asks for `<tool_call>` tags.

**The hypothesis:** It will struggle badly, but including it establishes a baseline for what raw 1-bit inference looks like on CPU, and sets the stage for when instruction-tuned BitNet models become available.

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

Everything is run 3 times with majority-vote aggregation to smooth out non-determinism.

## Results at a glance

| Model | Tool Calls | Valid Args | Avg Latency | Restraint |
|---|---|---|---|---|
| qwen2.5:3b | 7/9 | 7/7 | 3,927 ms | 1/2 |
| qwen2.5:1.5b | 5/9 | 5/5 | 2,886 ms | 2/2 |
| qwen2.5:0.5b | 6/9 | 6/6 | 1,487 ms | 2/2 |
| llama3.2:3b | 9/9 | 9/9 | 2,428 ms | 0/2 |
| smollm2:1.7b | 6/9 | 6/6 | 2,549 ms | 2/2 |
| bitnet-3B | 0/9 | 0/0 | 19,162 ms | 2/2* |

*\*BitNet's "restraint" is vacuous -- it never called any tool on any prompt because it couldn't produce coherent output.*

### The takeaways

**LLaMA 3.2 calls tools on everything.** 9/9 tool calls, 0/2 restraint. It called `search_files` when asked "What tools do you have access to?" and again when asked to write a Python script. It's a hammer that sees every prompt as a nail. If you need aggressive tool use and will filter false positives downstream, it's the most capable. If you need an agent that exercises judgment, it's the worst.

**The smaller models are more disciplined.** qwen2.5:1.5b, qwen2.5:0.5b, and smollm2:1.7b all scored 2/2 on restraint. They correctly identified that P5 and P9 don't need tools. This is counterintuitive -- you might expect bigger models to have better judgment, but the smaller models were more conservative by default, which happened to be the right call.

**smollm2 is the surprise performer.** 6/9 tool calls, 2/2 restraint, competitive latency. It matched the Qwen models on the metrics that matter and beat LLaMA on restraint. For a model that doesn't get much benchmark attention, it delivered.

**qwen2.5:3b fails the "weather in code" trap.** When asked to write a Python script about weather APIs, it called `get_weather` in 2 out of 3 runs. The word "weather" in the prompt was a stronger signal than the actual intent. The 1.5B version didn't make this mistake.

**BitNet is not ready for structured tasks.** Zero tool calls, incoherent output (word salad), ~20 seconds per prompt. This is entirely expected for a non-instruction-tuned base model. The 1.58-bit quantisation works for text generation in a completion sense, but without instruction tuning there's nothing to steer it toward tool-calling behavior. This benchmark should be re-run when instruction-tuned BitNet variants ship.

**P9 is the best discriminator prompt.** "Can you write a Python script that checks the weather using an API?" cleanly separates models that understand user intent from models that pattern-match on keywords. It's the single most useful prompt in the set for evaluating agent quality.

For the full data -- per-run breakdowns, latency tables, raw BitNet output samples -- see [REPORT.md](REPORT.md).

## Run it yourself

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- Python 3.10+
- ~10 GB free disk space for models
- For BitNet: `cmake`, `clang`, `clang++`, and ~12 GB additional disk space during build

### Quick start (Ollama models only)

If you just want to test the 5 Ollama models and skip BitNet:

```bash
# Pull the models
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama pull smollm2:1.7b

# Clone and set up
git clone <this-repo> && cd local-agent-bench
python -m venv .venv
source .venv/bin/activate
pip install ollama requests

# Comment out the bitnet-3B entry in ALL_MODELS and the
# start/stop_bitnet_server() calls in main(), then run:
python bench.py
```

### Full setup (including BitNet)

```bash
# 1. Pull Ollama models (same as above)
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama pull smollm2:1.7b

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
pip install --no-deps 3rdparty/llama.cpp/gguf-py  # workaround for sentencepiece build issue

# Run setup (downloads ~4GB model, builds llama.cpp)
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-3B -q i2_s

# If compile fails with a const-correctness error in ggml-bitnet-mad.cpp,
# change line 811 from "int8_t * y_col" to "const int8_t * y_col" and rebuild:
#   cmake --build build --config Release

# Verify the binary and model exist
ls build/bin/llama-server
ls models/bitnet_b1_58-3B/ggml-model-i2_s.gguf

# 3. Clone and run the benchmark
cd ~/projects
git clone <this-repo> && cd local-agent-bench
python -m venv .venv
source .venv/bin/activate
pip install ollama requests

# Update BITNET_DIR in bench.py if your BitNet path differs
python bench.py
```

The full run (6 models x 9 prompts x 3 runs = 162 inference calls) takes roughly 30-40 minutes on the hardware described above. Most of that time is BitNet -- the Ollama models finish in about 10 minutes total.

### Customising

The benchmark is a single file (`bench.py`). To add models, prompts, or adjust runs:

- **Add an Ollama model:** Add `{"name": "your-model:tag", "backend": "ollama"}` to `ALL_MODELS`
- **Add a prompt:** Append to `TEST_PROMPTS`. If it's a restraint prompt (correct answer is no tool call), add its 0-based index to `RESTRAINT_INDICES`
- **Change run count:** Edit `num_runs` in `main()`
- **Add tools:** Extend the `TOOLS` list and `TOOL_DISPATCH` dict. Update `BITNET_SYSTEM_PROMPT` if you want BitNet to know about them

## License

Do whatever you want with this. It's a benchmark script, not a product.
