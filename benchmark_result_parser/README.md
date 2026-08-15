# Benchmark Results Parser

A Python utility to parse benchmark JSON results and generate readable Markdown reports for tool-calling performance analysis.

## Overview

The benchmark parser transforms raw JSON benchmark output into a structured, human-readable Markdown report. It provides comprehensive statistics, per-run analysis, and tool-specific performance metrics, making it easier to evaluate model performance in tool-calling scenarios.

## Features

- 📊 **Comprehensive Statistics**: Overall metrics including total tool calls, validity rates, and latency analysis
- 📈 **Per-Run Analysis**: Detailed breakdown of each benchmark run with step-by-step results
- 🔧 **Tool-Specific Metrics**: Success rates and average latency for each tool type
- 📝 **Readable Format**: Clear tables, organized sections, and formatted content
- 🔍 **Response Content**: Shows model responses when tools weren't called for better debugging
- 📋 **Argument Details**: Displays the arguments used in each tool call
- 🚀 **Easy to Use**: Simple command-line interface

## Installation

The benchmark parser is included as part of the benchmark suite. No additional dependencies are required beyond Python 3.6+.

## Usage

### Basic Usage

```bash
python benchmark_parser.py <path_to_json_file>
