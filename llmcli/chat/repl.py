"""The interactive chat REPL.

Plain streaming chat by default; with `--rag` (or `/rag on`) every turn is
grounded on the documents llmcli.corpus has indexed for this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from llmcli.corpus.rag import RAG_SYSTEM, build_context, retrieve
from llmcli.corpus.store import Store
from llmcli.tools.args import Parents
from llmcli.tools.client import LLMClient
from llmcli.tools.workspace import Config

DEFAULT_SYSTEM = "You are a concise, accurate assistant."

HELP = """
  /rag on|off     toggle document retrieval for following turns
  /sources        show the chunks retrieved for the last question
  /system TEXT    replace the system prompt
  /reset          clear the conversation history
  /save FILE      write the transcript to FILE
  /help  /exit
"""


def cmd_chat(args, cfg: Config) -> None:
    cfg.require()
    client = LLMClient(cfg)
    store = Store(cfg.db)
    use_rag = args.rag
    system = args.system or cfg.system_prompt or (RAG_SYSTEM if use_rag else DEFAULT_SYSTEM)
    history: List[dict] = []
    last_hits: List[Tuple[float, str, str]] = []

    print(f"model {cfg.model}  rag {'on' if use_rag else 'off'}  (/help for commands)")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd, rest = cmd.lower(), rest.strip()
            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd == "/help":
                print(HELP)
            elif cmd == "/rag":
                use_rag = rest != "off"
                print(f"rag {'on' if use_rag else 'off'}")
            elif cmd == "/sources":
                for score, path, text in last_hits:
                    print(f"\n[{score:.3f}] {path}\n{text[:400]}...")
                if not last_hits:
                    print("(nothing retrieved yet)")
            elif cmd == "/system":
                system = rest or system
                print("system prompt updated")
            elif cmd == "/reset":
                history.clear()
                print("history cleared")
            elif cmd == "/save":
                target = Path(rest or "transcript.md").expanduser()
                target.write_text(
                    "\n\n".join(f"**{m['role']}**: {m['content']}" for m in history)
                )
                print(f"wrote {target}")
            else:
                print("unknown command - /help")
            continue

        if use_rag:
            last_hits = retrieve(client, store, line, args.top_k, args.min_score)
            context = build_context(last_hits)
            turn = (
                f"Context:\n{context}\n\nQuestion: {line}" if context else line
            )
        else:
            turn = line

        messages = [{"role": "system", "content": system}] + history[-2 * args.history_turns :]
        messages.append({"role": "user", "content": turn})
        reply = client.chat(
            messages, stream=not args.no_stream, temperature=args.temperature
        )
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": reply})
        if use_rag and last_hits:
            print("sources: " + ", ".join(sorted({Path(p).name for _, p, _ in last_hits})))

    store.close()


def register(sub, parents: Parents) -> None:
    p = sub.add_parser("chat", parents=[parents.common, parents.gen, parents.ret],
                       help="interactive chat")
    p.add_argument("--rag", action="store_true", help="ground answers on ingested docs")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--history-turns", type=int, default=8)
    p.set_defaults(func=cmd_chat)
