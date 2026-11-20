#!/usr/bin/env python3
"""
Nano-Voice Edge Assistant Entry Point.
Run voice assistant or interactive text loop on NVIDIA Jetson or local machine.
"""

import os
import sys
import argparse

# Ensure repo root is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assistant.pipeline import VoiceAssistantPipeline


def main():
    parser = argparse.ArgumentParser(description="Nano-Voice Edge AI Physical Assistant")
    parser.add_argument(
        "--model",
        type=str,
        default="scratch",
        choices=["scratch", "instruct"],
        help="LLM backend: 'scratch' (custom PyTorch weights) or 'instruct' (compact open instruct model)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="public",
        help="Profile directory: 'public' (sanitized demo) or 'private' (personal memory)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run in interactive text typing mode without requiring microphone hardware"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Audio input device index (run tools/check_env.py to inspect devices)"
    )
    args = parser.parse_args()

    profile_path = f"profiles/{args.profile}"
    if not os.path.exists(profile_path):
        print(f"[!] Profile directory '{profile_path}' does not exist. Defaulting to 'profiles/public'.")
        profile_path = "profiles/public"

    # If scratch model weights don't exist yet, run toy training automatically
    if args.model == "scratch" and not os.path.exists("checkpoints/model_final.pt"):
        print("[!] No scratch model checkpoint found. Running quick 30s toy training to build weights...")
        from tools.train_toy import main as run_toy_train
        run_toy_train()

    pipeline = VoiceAssistantPipeline(
        llm_type=args.model,
        profile_path=profile_path,
        device_id=args.device_id
    )

    if args.text_only:
        print("\n" + "=" * 60)
        print("💬 Running in interactive TEXT-ONLY mode.")
        print("   Type your questions below. Type 'exit' or 'quit' to end.")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                pipeline.process_query_text(user_input)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")
    else:
        pipeline.run_voice_loop()


if __name__ == "__main__":
    main()
