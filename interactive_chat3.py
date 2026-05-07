"""Terminal chat client for llama‑cpp models with sandboxed file I/O."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict

from llama_cpp import Llama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

#------
# Logging & console
#------
console = Console()
log = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
 """Initialise the root logger."""
 level = logging.DEBUG if verbose else logging.INFO
 logging.basicConfig(
     format="%(asctime)s %(levelname)s %(message)s",
     level=level,
 )


#------
# Filesystem sandbox configuration
#------
# Directory that the assistant is allowed to read/write.
# Can be overridden with the environment variable LLM_ALLOWED_ROOT.
ALLOWED_ROOT = Path(os.getenv("LLM_ALLOWED_ROOT", "data")).expanduser().resolve()


def _ensure_allowed(path: Path) -> Path:
 """
 Resolve *path* against ``ALLOWED_ROOT`` and verify that the result stays
 inside the sandbox.  Raises ``ValueError`` if the check fails.
 """
 full_path = (ALLOWED_ROOT / path).resolve()
 try:
     # Python 3.9+: Path.is_relative_to
     if not full_path.is_relative_to(ALLOWED_ROOT):  # type: ignore[attr-defined]
         raise ValueError
 except AttributeError:  # pragma: no cover – fallback for <3.9
     if not str(full_path).startswith(str(ALLOWED_ROOT)):
         raise ValueError
 return full_path


def read_file(rel_path: str) -> str:
    """Return the contents of *rel_path* (relative to ``ALLOWED_ROOT``)."""
    target = _ensure_allowed(Path(rel_path))
    with target.open("r", encoding="utf-8") as f:
        # Sanitize immediately upon reading
        return f.read().replace("<|", "< |")


def write_file(rel_path: str, content: str) -> None:
 """Write *content* to *rel_path* (relative to ``ALLOWED_ROOT``)."""
 target = _ensure_allowed(Path(rel_path))
 target.parent.mkdir(parents=True, exist_ok=True)
 with target.open("w", encoding="utf-8") as f:
     f.write(content)


#------
# Misc helpers
#------
DEFAULT_MODEL = Path(
 "/mnt/beegfs/cantilk/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
)
DEFAULT_CTX = 131_072
DEFAULT_REFRESH = 4

def sanitize_input(text: str) -> str:
    """
    Escape the special ``<|`` token anywhere it appears.
    Llama-CPP uses these for control tokens and will throw a ValueError 
    if they appear in the message content.
    """
    # This replaces all occurrences of '<|' with '< |'
    return text.replace("<|", "< |")


def maybe_multiline(first_line: str) -> str:
 """Enter a simple multiline mode when the user types \""" or /code."""
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
# Core chat session
#------
class ChatSession:
 def __init__(
     self,
     llm: Llama,
     system_prompt: str = "You are a helpful, concise AI assistant.",
 ) -> None:
     self.llm = llm
     self.history: List[Dict[str, str]] = [
         {"role": "system", "content": system_prompt}
     ]

 def add_user(self, text: str) -> None:
     self.history.append({"role": "user", "content": sanitize_input(text)})

 def add_assistant(self, text: str) -> None:
     self.history.append({"role": "assistant", "content": sanitize_input(text)})

 def stream_reply(self):
     """Yield (new_token, accumulated_text) while streaming the model."""
     full = ""
     stream = self.llm.create_chat_completion(messages=self.history, stream=True)
     for chunk in stream:
         delta = chunk["choices"][0]["delta"]
         if token := delta.get("content"):
             full += token
             yield token, full
     return full


#------
# REPL driver (now with /read and /write)
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

     #--------------
     # Special sandboxed file‑access commands
     #--------------
     if raw.startswith("/read "):
         rel_path = raw[len("/read ") :].strip()
         try:
             content = read_file(rel_path)
             console.print(f"[bold green]File '{rel_path}' contents:[/]\n{content}")
             # Feed the read content back to the model so it can reason about it
             session.add_user(f"[READ {rel_path}]\n{content}")
         except Exception as exc:  # pylint: disable=broad-except
             console.print(f"[bold red]Error reading '{rel_path}': {exc}[/]")
         continue

     if raw.startswith("/write "):
         # Syntax: /write <relative‑path>
         parts = raw.split(maxsplit=1)
         if len(parts) != 2:
             console.print("[bold red]Usage: /write <relative‑path>[/]")
             continue
         rel_path = parts[1].strip()
         console.print(
             "[bold cyan]--- WRITE MODE (type EOF on a line by itself to finish) ---[/]"
         )
         lines: List[str] = []
         while True:
             try:
                 line = input()
             except (EOFError, KeyboardInterrupt):
                 console.print("[bold red]Write cancelled[/]")
                 lines = []
                 break
             if line.strip() == "EOF":
                 break
             lines.append(line)
         if lines:
             try:
                 write_file(rel_path, "\n".join(lines))
                 console.print(
                     f"[bold green]Wrote {len(lines)} line(s) to '{rel_path}'[/]"
                 )
                 # Echo the write action back to the model (useful for reasoning)
                 session.add_user(f"[WRITE {rel_path}]\n" + "\n".join(lines))
             except Exception as exc:  # pylint: disable=broad-except
                 console.print(f"[bold red]Error writing '{rel_path}': {exc}[/]")
         continue

     #--------------
     # Normal chat flow
     #--------------
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
 p.add_argument(
     "--model_path",
     type=Path,
     default=DEFAULT_MODEL,
     help="Path to the GGUF model.",
 )
 p.add_argument(
     "--n_threads",
     type=int,
     default=16,
     help="CPU threads.",
 )
 p.add_argument(
     "--n_gpu_layers",
     type=int,
     default=-1,
     help="GPU layers (‑1 = auto).",
 )
 p.add_argument(
     "-v",
     "--verbose",
     action="store_true",
     help="Enable debug logging.",
 )
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

 console.print(
     "[bold green]LLama‑CPP Chat[/] – type /code or \"\"\" for multiline, "
     "'clear' to reset, '/read <path>' or '/write <path>'."
 )
 repl(ChatSession(llm))


if __name__ == "__main__":
 main()

