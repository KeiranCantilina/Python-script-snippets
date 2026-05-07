"""
Autonomous Sandboxed LLM Agent
Allows an LLM to read/write files within a restricted 'data' directory.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Generator, Any

from llama_cpp import Llama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

# ------
# Configuration & Sandbox
# ------
console = Console()
ALLOWED_ROOT = Path(os.getenv("LLM_ALLOWED_ROOT", "data")).expanduser().resolve()
DEFAULT_CTX = 131_072

def _ensure_allowed(path: Path) -> Path:
    """Verify path is within sandbox; raises ValueError if outside."""
    full_path = (ALLOWED_ROOT / path).resolve()
    if not full_path.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"Access denied: {path} is outside sandbox.")
    return full_path

def read_file(rel_path: str) -> str:
    """Read file content and sanitize special tokens."""
    target = _ensure_allowed(Path(rel_path))
    with target.open("r", encoding="utf-8") as f:
        return f.read().replace("<|", "< |")

def write_file(rel_path: str, content: str) -> str:
    """Write content to file and return status."""
    target = _ensure_allowed(Path(rel_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote to {rel_path}"

def sanitize_input(text: str) -> str:
    """Prevent ValueError by escaping special control tokens."""
    return text.replace("<|", "< |")

# ------
# Tool Definitions
# ------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"rel_path": {"type": "string"}},
                "required": ["rel_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["rel_path", "content"],
            },
        },
    }
]

# ------
# Agent Logic
# ------
class AgentSession:
    def __init__(self, llm: Llama):
        self.llm = llm
        self.history: List[Dict[str, Any]] = [{
            "role": "system",
            "content": "You are a file-manager agent. Use read_file and write_file tools to help the user."
        }]

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": sanitize_input(text)})

    def run(self) -> str:
        """The recursive agent loop for tool execution."""
        while True:
            response = self.llm.create_chat_completion(
                messages=self.history,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            message = response["choices"][0]["message"]
            self.history.append(message)

            if not message.get("tool_calls"):
                return message.get("content") or ""

            for tool_call in message["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                
                console.print(f"[bold yellow]⚙️ Executing {fn_name}({args.get('rel_path')})...[/]")
                
                try:
                    if fn_name == "read_file":
                        res = read_file(args["rel_path"])
                    elif fn_name == "write_file":
                        res = write_file(args["rel_path"], args["content"])
                    else:
                        res = "Error: Unknown tool."
                except Exception as e:
                    res = f"Error: {str(e)}"

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": res
                })

# ------
# CLI Driver
# ------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Path to GGUF model")
    args = parser.parse_args()

    ALLOWED_ROOT.mkdir(exist_ok=True)
    console.print(f"[bold cyan]Sandbox active at:[/] {ALLOWED_ROOT}")
    
    llm = Llama(model_path=str(args.model), n_ctx=DEFAULT_CTX, n_gpu_layers=-1)
    session = AgentSession(llm)

    while True:
        try:
            query = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.lower() in {"exit", "clear"}:
            if query.lower() == "clear": 
                session = AgentSession(llm)
                console.clear()
            else: break
            continue

        session.add_user(query)
        with console.status("[bold green]Agent thinking..."):
            answer = session.run()
        
        console.print(Markdown(answer))

if __name__ == "__main__":
    main()