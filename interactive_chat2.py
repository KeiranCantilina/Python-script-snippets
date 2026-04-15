"""Terminal chat client for llama‑cpp models."""

from __future__ import annotations
import argparse
import logging
import re
import readline
from pathlib import Path
from typing import List, Dict

from llama_cpp import Llama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

#------
# Constants & helpers
#------
DEFAULT_MODEL = Path("/mnt/beegfs/cantilk/gpt-oss-120b-MXFP4-00001-of-00002.gguf")
DEFAULT_CTX = 131_072
DEFAULT_REFRESH = 4
console = Console()
log = logging.getLogger(__name__)

def configure_logging(verbose: bool) -> None:
 level = logging.DEBUG if verbose else logging.INFO
 logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=level)

def sanitize_input(text: str) -> str:
 return re.sub(r"<\|", "< |", text)

def maybe_multiline(first_line: str) -> str:
 trigger = first_line.strip()
 if trigger not in {'"""', "/code"}:
     return first_line
 console.print("[bold cyan]--- MULTILINE MODE (type EOF to finish) ---[/]")
 lines = []
 while True:
     line = input()
     if line.strip() == "EOF":
         break
     lines.append(line)
 console.print("[bold cyan]--- END OF BLOCK ---[/]")
 return "\n".join(lines)

#------
# Core classes
#------
class ChatSession:
 def __init__(self, llm: Llama,
              system_prompt: str = "You are a helpful, concise AI assistant.") -> None:
     self.llm = llm
     self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

 def add_user(self, text: str) -> None:
     self.history.append({"role": "user", "content": sanitize_input(text)})

 def add_assistant(self, text: str) -> None:
     self.history.append({"role": "assistant", "content": sanitize_input(text)})

 def stream_reply(self):
     full = ""
     stream = self.llm.create_chat_completion(messages=self.history, stream=True)
     for chunk in stream:
         delta = chunk["choices"][0]["delta"]
         if token := delta.get("content"):
             full += token
             yield token, full
     return full

#------
# REPL driver
#------
def repl(session: ChatSession) -> None:
 while True:
     try:
         raw = input("\n>>> ")
     except (EOFError, KeyboardInterrupt):
         break

     if not raw.strip():
         continue
     if raw.lower() in {"bye", "exit"}:
         break
     if raw.lower() == "clear":
         session.__init__(session.llm)
         console.clear()
         continue

     user_msg = maybe_multiline(raw)
     session.add_user(user_msg)

     with Live(console=console, refresh_per_second=DEFAULT_REFRESH) as live:
         for _, acc in session.stream_reply():
             live.update(Markdown(acc))

     # The generator already produced the final text in `acc`
     session.add_assistant(acc)

#------
# Argument parsing & entry point
#------
def parse_args() -> argparse.Namespace:
 p = argparse.ArgumentParser(description="Chat with a Llama‑CPP model.")
 p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL,
                help="Path to the GGUF model.")
 p.add_argument("--n_threads", type=int, default=16,
                help="CPU threads.")
 p.add_argument("--n_gpu_layers", type=int, default=-1,
                help="GPU layers (‑1 = auto).")
 p.add_argument("-v", "--verbose", action="store_true",
                help="Enable debug logging.")
 return p.parse_args()

def load_model(params: dict) -> Llama:
 try:
     return Llama(**params)
 except Exception as exc:  # pylint: disable=broad-except
     log.exception("Unable to load model")
     raise SystemExit(1) from exc

def main() -> None:
 args = parse_args()
 configure_logging(args.verbose)

 model_params = {
     "model_path": str(args.model_path),
     "n_ctx": DEFAULT_CTX,
     "n_threads": args.n_threads,
     "verbose": args.verbose,
     "n_gpu_layers": args.n_gpu_layers,
 }
 console.print("Loading model. Please wait...\n")
 llm = load_model(model_params)
 console.print("[bold green]LLama‑CPP Chat[/] – type /code or \"\"\" for multiline, 'clear' to reset.")
 repl(ChatSession(llm))

if __name__ == "__main__":
 main()

