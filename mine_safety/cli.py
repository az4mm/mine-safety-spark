"""Command-line entry points. Run from the cloned repository."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mine safety analytics with Apache Spark")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("generate", help="Create explicitly synthetic camera observations and expected results")
    p.add_argument("--output", default="data/demo")
    p.add_argument("--seconds", type=int, default=120)
    for command in ("batch", "stream"):
        p = commands.add_parser(command)
        p.add_argument("--input", default="data/demo/events.jsonl" if command == "batch" else "data/runtime/inbox")
        p.add_argument("--output", default="outputs/current")
        p.add_argument("--master", default="local[2]")
        if command == "stream":
            p.add_argument("--checkpoint", default="checkpoints/current")
            p.add_argument("--trigger", type=int, default=2)
            p.add_argument("--available-now", action="store_true")
    p = commands.add_parser("replay", help="Write synthetic JSON microbatches for a running Spark stream")
    p.add_argument("--output", default="data/runtime/inbox")
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--delay", type=float, default=2)
    p.add_argument("--batch-seconds", type=int, default=5)
    p = commands.add_parser("serve", help="Start local dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--output", default="outputs/current")
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .simulation import prepare_demo
        print(json.dumps(prepare_demo(args.output, args.seconds), indent=2))
    elif args.command == "replay":
        from .simulation import replay
        print(json.dumps(replay(args.output, args.seconds, args.batch_seconds, args.delay), indent=2))
    elif args.command == "batch":
        from .spark_jobs import run_batch
        print(json.dumps(run_batch(args.input, args.output, master=args.master), indent=2, default=str))
    elif args.command == "stream":
        from .spark_jobs import run_stream
        Path(args.input).mkdir(parents=True, exist_ok=True)
        query = run_stream(args.input, args.output, args.checkpoint, master=args.master, trigger_seconds=args.trigger, available_now=args.available_now)
        if query is not None:
            query.awaitTermination()
    elif args.command == "serve":
        import os
        import uvicorn
        os.environ["MINE_SAFETY_OUTPUT"] = str(Path(args.output).resolve())
        uvicorn.run("mine_safety.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
