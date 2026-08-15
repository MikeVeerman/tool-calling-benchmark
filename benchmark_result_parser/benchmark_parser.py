import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
from pathlib import Path

def parse_benchmark_results(json_file_path: str):
    """
    Parse benchmark JSON results and generate a readable Markdown report.
    
    Args:
        json_file_path: Path to the JSON file containing benchmark results
    
    Returns:
        str: Path to the generated Markdown file
    """
    
    
    
    output_md_path = f"benchmark_result_{Path(json_file_path).stem}.md"

    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark input file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {json_file_path}: {e.msg}", e.doc, e.pos)
    
    # Initialize markdown content
    md_content = []
    
    # Header
    md_content.append("# Tool Calling Benchmark Results\n")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("---\n")
    
    # Model Information
    md_content.append("## Model Information\n")
    md_content.append(f"- **Model Name:** `{data.get('model_name', 'N/A')}`")
    md_content.append(f"- **Backend:** {data.get('model_info', {}).get('backend', 'N/A')}")
    md_content.append(f"- **Origin:** {data.get('model_info', {}).get('origin', 'N/A')}")
    md_content.append(f"- **Benchmark Version:** `{data.get('bench_version', 'N/A')}`")
    md_content.append(f"- **Number of Runs:** {data.get('num_runs', 0)}")
    md_content.append(f"- **Timestamp:** {data.get('timestamp', 'N/A')}\n")
    md_content.append("---\n")
    
    # Overall Statistics
    md_content.append("## Overall Statistics\n")
    
    all_runs = data.get('runs', [])
    total_runs = len(all_runs)
    
    # Calculate overall statistics
    total_tool_calls = 0
    total_valid_args = 0
    total_latency = 0
    run_stats = []
    
    for run_idx, run in enumerate(all_runs, 1):
        run_tool_calls = 0
        run_valid_args = 0
        run_latency = 0
        run_results = []
        
        for result in run:
            if result.get('tool_called', False):
                run_tool_calls += 1
                run_latency += result.get('latency_ms', 0)
                if result.get('valid_args', False):
                    run_valid_args += 1
                run_results.append({
                    'tool': result.get('tool_name', 'unknown'),
                    'valid': result.get('valid_args', False),
                    'latency': result.get('latency_ms', 0)
                })
        
        # Calculate per-run statistics
        total_tool_calls += run_tool_calls
        total_valid_args += run_valid_args
        total_latency += run_latency
        run_stats.append({
            'run': run_idx,
            'tool_calls': run_tool_calls,
            'valid_args': run_valid_args,
            'total_latency': run_latency,
            'avg_latency': run_latency / run_tool_calls if run_tool_calls > 0 else 0,
            'success_rate': (run_valid_args / run_tool_calls * 100) if run_tool_calls > 0 else 0
        })
    
    # Summary statistics
    avg_tool_calls = total_tool_calls / total_runs if total_runs > 0 else 0
    avg_latency = total_latency / total_tool_calls if total_tool_calls > 0 else 0
    overall_success_rate = (total_valid_args / total_tool_calls * 100) if total_tool_calls > 0 else 0
    
    md_content.append(f"- **Total Tool Calls (all runs):** {total_tool_calls}")
    md_content.append(f"- **Total Valid Arguments:** {total_valid_args}")
    md_content.append(f"- **Average Tool Calls per Run:** {avg_tool_calls:.2f}")
    md_content.append(f"- **Average Latency per Tool Call:** {avg_latency:.2f} ms")
    md_content.append(f"- **Overall Argument Validity Rate:** {overall_success_rate:.2f}%\n")
    
    # Run Details Table
    md_content.append("### Run Statistics\n")
    md_content.append("| Run | Tool Calls | Valid Arguments | Success Rate | Total Latency (ms) | Avg Latency (ms) |")
    md_content.append("|-----|------------|-----------------|--------------|-------------------|-----------------|")
    for stats in run_stats:
        md_content.append(
            f"| {stats['run']} | {stats['tool_calls']} | {stats['valid_args']} | "
            f"{stats['success_rate']:.1f}% | {stats['total_latency']} | {stats['avg_latency']:.1f} |"
        )
    
    md_content.append("\n---\n")
    
    # Per-Run Detailed Results
    md_content.append("## Detailed Run Results\n")
    
    for run_idx, run in enumerate(all_runs, 1):
        md_content.append(f"### Run #{run_idx}\n")
        
        # Count tool calls for this run
        tool_calls = [r for r in run if r.get('tool_called', False)]
        no_tool_calls = [r for r in run if not r.get('tool_called', False)]
        
        md_content.append(f"**Tool Calls:** {len(tool_calls)}")
        md_content.append(f"**No Tool Calls:** {len(no_tool_calls)}")
        md_content.append(f"**Total Steps:** {len(run)}\n")
        
        # Tool Usage Summary
        if tool_calls:
            md_content.append("#### Tool Calls Summary\n")
            md_content.append("| Step | Tool | Arguments Valid? | Latency (ms) | Arguments |")
            md_content.append("|------|------|------------------|--------------|-----------|")
            
            for i, result in enumerate(run, 1):
                if result.get('tool_called', False):
                    tool_name = result.get('tool_name', 'unknown')
                    valid_args = "✅" if result.get('valid_args') else "❌"
                    latency = result.get('latency_ms', 0)
                    
                    # Extract arguments from all_tool_calls
                    args_str = ""
                    all_calls = result.get('all_tool_calls', [])
                    if all_calls:
                        args = all_calls[0].get('arguments', {})
                        args_str = json.dumps(args, ensure_ascii=False)
                    
                    md_content.append(f"| {i} | `{tool_name}` | {valid_args} | {latency} | `{args_str}` |")
                else:
                    # Show responses without tool calls
                    raw_content = result.get('raw_content', '')
                    md_content.append(f"| {i} | *No Tool Call* | - | {result.get('latency_ms', 0)} | *{raw_content[:50]}{'...' if len(raw_content) > 50 else ''}* |")
            
            md_content.append("")
        
        # No Tool Call Responses
        if no_tool_calls:
            md_content.append("#### Responses Without Tool Calls\n")
            for i, result in enumerate(run, 1):
                if not result.get('tool_called', False):
                    content = result.get('raw_content', 'No content available')
                    md_content.append(f"**Step {i}** (Latency: {result.get('latency_ms', 0)} ms)")
                    md_content.append(f"> {content}\n")
        
        md_content.append("---\n")
    
    # Tool Statistics
    md_content.append("## Tool Usage Statistics\n")
    
    tool_stats = defaultdict(lambda: {'count': 0, 'valid': 0, 'latency': 0})
    
    for run in all_runs:
        for result in run:
            if result.get('tool_called', False):
                tool_name = result.get('tool_name', 'unknown')
                tool_stats[tool_name]['count'] += 1
                tool_stats[tool_name]['latency'] += result.get('latency_ms', 0)
                if result.get('valid_args', False):
                    tool_stats[tool_name]['valid'] += 1
    
    md_content.append("| Tool | Total Calls | Valid Arguments | Success Rate | Avg Latency (ms) |")
    md_content.append("|------|-------------|-----------------|--------------|------------------|")
    for tool_name, stats in sorted(tool_stats.items()):
        success_rate = (stats['valid'] / stats['count'] * 100) if stats['count'] > 0 else 0
        avg_latency = stats['latency'] / stats['count'] if stats['count'] > 0 else 0
        md_content.append(
            f"| `{tool_name}` | {stats['count']} | {stats['valid']} | "
            f"{success_rate:.1f}% | {avg_latency:.1f} |"
        )
    
    # Write to file 
    try:
        
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
    except IOError as e:
        raise IOError(f"Failed to write output file {output_md_path}: {e}")
    
    print(f"✅ Benchmark report generated: {output_md_path}")
    return output_md_path

# Example usage
if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        try:
            output_file = parse_benchmark_results(json_file)
        except FileNotFoundError as e :
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    else:
     print("Error: No file were given.")