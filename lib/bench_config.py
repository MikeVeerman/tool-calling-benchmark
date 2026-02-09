"""Shared constants for the local LLM tool-calling benchmark."""

# ---------------------------------------------------------------------------
# Tool definitions (Ollama format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Antwerp'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files, e.g. '*.py'",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with attendees at a given time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting title",
                    },
                    "time": {
                        "type": "string",
                        "description": "Meeting time in ISO 8601 format",
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses",
                    },
                },
                "required": ["title", "time"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Test prompts – from obvious to ambiguous to trick to harder
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # P1 – obvious single-tool
    "What's the weather in Antwerp?",
    # P2 – obvious different tool
    "Find all Python files in the project.",
    # P3 – requires multiple args
    "Schedule a meeting called 'Sprint Review' for 2025-02-10T14:00:00 with alice@co.com and bob@co.com.",
    # P4 – ambiguous, could use tool or not
    "I'm heading to Brussels tomorrow, anything I should know?",
    # P5 – trick / meta question, should NOT call a tool
    "What tools do you have access to?",
    # P6 – multi-step reasoning (requires chaining context not available)
    "What's the weather in the city where we have our next sprint review?",
    # P7 – noisy parameter extraction
    "Oh hey, could you maybe like set up a meeting — 'Q3 Roadmap' — for next Tuesday at 3pm? I think dave@co.com and maybe susan@co.com should come",
    # P8 – adversarial: asks for TWO tools at once
    "Search for all files matching '*.py' and also tell me the weather in Paris.",
    # P9 – tool-adjacent trick, should NOT call a tool
    "Can you write a Python script that checks the weather using an API?",
    # P10 – implicit reasoning: cycling decision depends on weather, "weather" never mentioned
    "I have a meeting with a client in Bruges next Thursday. Should I take the train or cycle?",
    # P11 – negation: explicitly says "don't check weather"
    "Don't check the weather in Antwerp, just find me the quarterly report.",
    # P12 – redundant tool trap: weather already provided, just schedule
    "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?",
]

# Indices of prompts where the correct behavior is to NOT call a tool
# P5 (idx 4): meta question about tools
# P9 (idx 8): asking to write code, not to call a tool
RESTRAINT_INDICES = {4, 8}

# Indices of prompts where calling a valid tool is clearly correct (for Agent Score)
# P1, P2, P3, P4, P6, P7, P8 are clear tool-call prompts; P10, P11, P12 are hard prompts
TOOL_CALL_INDICES = {0, 1, 2, 3, 5, 6, 7, 9, 10, 11}  # 10 prompts

# P10-P12: expected correct tool for each hard prompt
EXPECTED_TOOLS = {
    9: "get_weather",       # P10: cycling depends on weather (implicit reasoning)
    10: "search_files",     # P11: find the report (negation)
    11: "schedule_meeting", # P12: schedule the meeting (context awareness)
}

# P10-P12: tools that are WRONG (worse than not calling at all)
WRONG_TOOL_MAP = {
    9: {"schedule_meeting"},  # P10: meeting already exists
    10: {"get_weather"},      # P11: explicitly told "don't"
    11: {"get_weather"},      # P12: weather already provided
}

HARD_PROMPT_INDICES = {9, 10, 11}  # P10, P11, P12

# ---------------------------------------------------------------------------
# Backend display names
# ---------------------------------------------------------------------------

BACKEND_DISPLAY = {
    "ollama":     ("Ollama",     "native-tools"),
    "ollama_raw": ("Ollama",     "raw-schema"),
    "bitnet":     ("bitnet.cpp", "openai-compat"),
    "llamacpp":   ("llama.cpp",  "openai-compat"),
}

P8_REQUIRED_TOOLS = {"search_files", "get_weather"}


def get_backend_display(model_info: dict) -> tuple[str, str]:
    """Return (backend_name, mode) for display purposes."""
    return BACKEND_DISPLAY[model_info["backend"]]


# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

ALL_MODELS = [
    {"name": "qwen2.5:3b",      "backend": "ollama",  "origin": "CN"},
    {"name": "qwen2.5:1.5b",    "backend": "ollama",  "origin": "CN"},
    {"name": "qwen2.5:0.5b",    "backend": "ollama",  "origin": "CN"},
    {"name": "llama3.2:3b",     "backend": "ollama",  "origin": "US"},
    {"name": "smollm2:1.7b",    "backend": "ollama",  "origin": "US"},
    {"name": "ministral-3:3b",  "backend": "ollama",  "origin": "FR"},
    {"name": "deepseek-r1:1.5b","backend": "ollama_raw",  "origin": "CN"},
    {"name": "gemma3:1b",       "backend": "ollama_raw",  "origin": "US"},
    {"name": "phi4-mini:3.8b",  "backend": "ollama_raw",  "origin": "US"},
    {"name": "bitnet-3B",       "backend": "bitnet",  "origin": "US/1bit",
     "model_path": "/home/mike/projects/bitnet/models/bitnet_b1_58-3B/ggml-model-i2_s.gguf"},
    {"name": "bitnet-2B-4T",    "backend": "bitnet",  "origin": "US/1bit",
     "model_path": "/home/mike/projects/bitnet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"},
    # Round 2 models (community-requested)
    {"name": "qwen3:0.6b",      "backend": "ollama",  "origin": "CN"},
    {"name": "qwen3:1.7b",      "backend": "ollama",  "origin": "CN"},
    {"name": "qwen3:4b",        "backend": "ollama",  "origin": "CN"},
    {"name": "functiongemma",    "backend": "ollama",  "origin": "US"},
    {"name": "granite3.3:2b",   "backend": "ollama",  "origin": "US"},
    {"name": "llama3.2:1b",     "backend": "ollama",  "origin": "US"},
    {"name": "lfm2.5:1.2b",    "backend": "llamacpp", "origin": "US",
     "model_id": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"},
]

# Sub-2B models for the "edge agent" mini leaderboard
EDGE_MODELS = {"qwen2.5:0.5b", "qwen2.5:1.5b", "smollm2:1.7b", "deepseek-r1:1.5b", "gemma3:1b", "bitnet-2B-4T",
               "qwen3:0.6b", "qwen3:1.7b", "functiongemma", "llama3.2:1b", "lfm2.5:1.2b"}
