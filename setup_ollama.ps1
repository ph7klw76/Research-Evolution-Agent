# ============================================================
# Ollama + Llama 3.1 8B Setup Script
# Research Evolution Agent
# ============================================================
# Run this in PowerShell (as Administrator recommended):
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_ollama.ps1
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Research Evolution Agent — Ollama Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Install Ollama ────────────────────────────────────
Write-Host "[1/5] Installing Ollama..." -ForegroundColor Yellow
try {
    irm https://ollama.com/install.ps1 | iex
    Write-Host "      Ollama installed successfully." -ForegroundColor Green
} catch {
    Write-Host "      ERROR during Ollama install: $_" -ForegroundColor Red
    exit 1
}

# ── Step 2: Verify version ───────────────────────────────────
Write-Host ""
Write-Host "[2/5] Verifying Ollama version..." -ForegroundColor Yellow
$version = & ollama --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "      $version" -ForegroundColor Green
} else {
    Write-Host "      WARNING: Could not verify Ollama version. Try restarting your terminal." -ForegroundColor Red
}

# ── Step 3: Pull Llama 3.1 8B ────────────────────────────────
Write-Host ""
Write-Host "[3/5] Pulling llama3.1:8b (~4.7 GB — this may take a while)..." -ForegroundColor Yellow
& ollama pull llama3.1:8b
if ($LASTEXITCODE -eq 0) {
    Write-Host "      Model pulled successfully." -ForegroundColor Green
} else {
    Write-Host "      ERROR: Failed to pull model. Make sure Ollama service is running." -ForegroundColor Red
    exit 1
}

# ── Step 4: Confirm model list ───────────────────────────────
Write-Host ""
Write-Host "[4/5] Confirming available models..." -ForegroundColor Yellow
& ollama list

# ── Step 5: Quick CLI test ───────────────────────────────────
Write-Host ""
Write-Host "[5/5] Running a quick CLI smoke test..." -ForegroundColor Yellow
Write-Host "      (Sending prompt: 'Say: Research Evolution Agent is ready.')" -ForegroundColor Gray
$response = echo "Say exactly: Research Evolution Agent is ready." | & ollama run llama3.1:8b 2>&1
Write-Host ""
Write-Host "      Model response:" -ForegroundColor Cyan
Write-Host "      $response" -ForegroundColor White

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete! Now run the Python test:" -ForegroundColor Cyan
Write-Host "    python test_ollama.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
