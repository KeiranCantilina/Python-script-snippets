import argparse
import os
import sys
import readline

from llama_cpp import Llama
from termcolor import cprint
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

console = Console()

#model_path = "/mnt/beegfs/cantilk/.cache/huggingface/hub/models--lmstudio-community--Qwen3-30B-A3B-GGUF/snapshots/98773a4c5f7f32199dd3a2a484deff0864a8d6f2/Qwen3-30B-A3B-Q4_K_M.gguf"

#model_path = "/mnt/beegfs/cantilk/.cache/huggingface/hub/models--lmstudio-community--Qwen3.5-122B-A10B-GGUF/snapshots/a5ba02062d2fd26c1f17000571ffc86e7abf4c92/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf"

model_path = "/mnt/beegfs/cantilk/gpt-oss-120b-MXFP4-00001-of-00002.gguf"

print_bot = lambda x: cprint(x, 'green', end='')

def main(model_path, verbose=False, n_threads=16, n_gpu_layers=-1):
    params = {
        "model_path": model_path,
        "n_ctx": 131072, # Adjust based on your VRAM/RAM
        "n_threads": int(n_threads),
        "verbose": verbose,
        "n_gpu_layers": int(n_gpu_layers),
    }

    llm = Llama(**params)
   # os.system("clear")

    # Use a list of dictionaries for clean state management
    history = [
        {"role": "system", "content": "You are a helpful, concise AI assistant."}
    ]

    print("Chat session started. Terminate queries with \"/no_think\" to disable thinking.")

    while True:
        try:
            user_input = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        if user_input.lower() == "clear":
            history = [{"role": "system", "content": "You are a helpful assistant."}]
            print("=== History Cleared! ===")
            continue

        if user_input.lower() in ["bye", "exit"]:
            break

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Use create_chat_completion to prevent character "echoing"
        stream = llm.create_chat_completion(
            messages=history,
           # max_tokens=1024,
#	    chat_template_kwargs={"enable_thinking": False},
#            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=True,
        )

        full_response = ""
        with Live(console=console, refresh_per_second=10) as live:
                for chunk in stream:
                        if "content" in chunk["choices"][0]["delta"]:
                                token = chunk["choices"][0]["delta"]["content"]
                                full_response += token
                                # print_bot(token)
                                # sys.stdout.flush()
                                live.update(Markdown(full_response))


        # Add assistant response to history
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=model_path)
    parser.add_argument("--n_threads", default=16, type=int)
    parser.add_argument("--n_gpu_layers", default=-1, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    main(args.model_path, args.verbose, args.n_threads, args.n_gpu_layers)
