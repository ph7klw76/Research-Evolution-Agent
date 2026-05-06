"""
test_ollama.py — Research Evolution Agent
==========================================
Tests the local Ollama server with llama3.1:8b using the Python client.

Run after setup_ollama.ps1:
    pip install ollama
    python test_ollama.py
"""

import sys

# ── 1. Check ollama package is installed ──────────────────────
try:
    import ollama
    print("[OK] ollama Python package imported successfully.")
except ImportError:
    print("[ERROR] ollama package not found. Run: pip install ollama")
    sys.exit(1)

# ── 2. List available models ──────────────────────────────────
print("\n[INFO] Models available on this machine:")
try:
    models = ollama.list()
    model_names = [m.model for m in models.models]
    if model_names:
        for name in model_names:
            print(f"       - {name}")
    else:
        print("       (none found — did you run 'ollama pull llama3.1:8b'?)")
except Exception as e:
    print(f"[ERROR] Could not list models: {e}")
    print("        Is the Ollama service running? Try: ollama serve")
    sys.exit(1)

# ── 3. Send a test prompt ─────────────────────────────────────
MODEL = "llama3.1:8b"
PROMPT = (
    'Please respond with exactly this sentence and nothing else: '
    '"Research Evolution Agent is ready."'
)

print(f'\n[INFO] Sending test prompt to {MODEL}...')
print(f'       Prompt: "{PROMPT}"')
print()

try:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    reply = response.message.content.strip()
    print("=" * 56)
    print(f"  Model reply: {reply}")
    print("=" * 56)

    # ── 4. Pass / fail check ──────────────────────────────────
    if "Research Evolution Agent is ready" in reply:
        print("\n[PASS] Model responded correctly. Ollama is working!")
    else:
        print("\n[WARN] Unexpected reply, but the model did respond.")
        print("       Check the output above.")

except ollama.ResponseError as e:
    print(f"[ERROR] Ollama API error: {e}")
    print(f"        Make sure '{MODEL}' is pulled: ollama pull {MODEL}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    sys.exit(1)

print("\n[DONE] test_ollama.py finished successfully.")
