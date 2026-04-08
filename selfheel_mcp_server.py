"""
self_healing_mcp_server_v2.py
==============================
Self-Healing MCP Server — Three Layer Intelligence Stack
---------------------------------------------------------
Layer 1 — Sentence Transformers  : Fast offline semantic matching of known errors
Layer 2 — GitHub Copilot API     : AI reasoning for unknown/complex errors
Layer 3 — failure_history.json   : Memory that learns what worked over time

Usage:
    Set GITHUB_TOKEN environment variable or update config below
    python self_healing_mcp_server_v2.py

Copilot Agent Prompts:
    "Run bash process daily A1"
    "Run bash process monthly B1"
    "Show failure history for daily_process1.bat"
    "Show all failure history"
    "Clear failure history for daily_process1.bat"
    "List all processes"
"""

from mcp.server.fastmcp import FastMCP
import subprocess
import os
import json
import time
import traceback
import requests
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Sentence Transformers — lazy loaded on first use
_sentence_model = None

mcp = FastMCP("self-healing-bash-runner-v2")


# -- Config --------------------------------------------------------------------
SCRIPTS_DIR           = "/workspaces/codespaces-demo/batch/scripts"
OUTPUT_DIR            = "/workspaces/codespaces-demo/batch/output"
CONFIG_FILE           = "/workspaces/codespaces-demo/batch/scripts/process_config.json"
LOG_FILE              = "/workspaces/codespaces-demo/batch/output/process_log.txt"
FAILURE_HISTORY       = "/workspaces/codespaces-demo/batch/output/failure_history.json"


MAX_RETRIES           = 3        # Max retry attempts per script
RETRY_DELAY_SECS      = 5        # Seconds between retries
TIMEOUT_SECS          = 30       # Script execution timeout
SIMILARITY_THRESHOLD  = 0.75     # Min cosine similarity to trust a match (0-1)

# GitHub Copilot API
GITHUB_TOKEN          = os.environ.get("GITHUB_TOKEN", "")   # set env var or paste token here
COPILOT_API_URL       = "https://api.githubcopilot.com/chat/completions"
COPILOT_MODEL         = "gpt-4o"
COPILOT_TIMEOUT       = 30       # seconds to wait for Copilot response
# -----------------------------------------------------------------------------


# -- Known Seed Patterns (Layer 1 seed data) -----------------------------------
# These bootstrap the semantic matcher before failure_history.json has entries.
# Add more as you discover new errors in production.
SEED_PATTERNS = [
    {
        "error_text" : "Access is denied",
        "fix_type"   : "command",
        "fix"        : "icacls {script_path} /grant Everyone:F",
        "description": "Permission denied on script or output file"
    },
    {
        "error_text" : "The system cannot find the path specified",
        "fix_type"   : "mkdir",
        "fix"        : "{output_dir}",
        "description": "Output directory does not exist"
    },
    {
        "error_text" : "The process cannot access the file because it is being used by another process",
        "fix_type"   : "wait",
        "fix"        : "10",
        "description": "File locked by another process"
    },
    {
        "error_text" : "No such file or directory",
        "fix_type"   : "log_only",
        "fix"        : "Input file missing - cannot auto-fix",
        "description": "Required input file is missing"
    },
    {
        "error_text" : "There is not enough space on the disk",
        "fix_type"   : "log_only",
        "fix"        : "Disk full - manual intervention required",
        "description": "Disk space exhausted"
    },
    {
        "error_text" : "script timed out execution exceeded limit",
        "fix_type"   : "wait",
        "fix"        : "15",
        "description": "Script execution timed out"
    },
    {
        "error_text" : "network path not found unreachable connection refused",
        "fix_type"   : "wait",
        "fix"        : "20",
        "description": "Network connectivity issue"
    },
    {
        "error_text" : "batch file syntax error unexpected token",
        "fix_type"   : "log_only",
        "fix"        : "Syntax error in batch file - manual fix required",
        "description": "Batch file has a syntax error"
    },
]
# -----------------------------------------------------------------------------


# =============================================================================
# UTILITIES
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Write timestamped log to file and stdout. CP1252-safe — no emojis."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    # Print safely to console — replace any unencodable chars
    print(line.encode("cp1252", errors="replace").decode("cp1252"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Write log file as UTF-8 so full content is preserved
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_failure_history() -> dict:
    """Load failure_history.json. Returns empty dict if not found."""
    try:
        if os.path.exists(FAILURE_HISTORY):
            with open(FAILURE_HISTORY, "r") as f:
                return json.load(f)
    except Exception as e:
        log(f"Could not load failure history: {e}", "WARNING")
    return {}


def save_failure_history(history: dict):
    """Persist failure_history.json."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(FAILURE_HISTORY, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log(f"Could not save failure history: {e}", "WARNING")


def record_failure(script_name: str, error: str, exit_code: int,
                   fix_applied: str, layer_used: str,
                   attempt: int, success_after_fix: bool):
    """
    Append a failure record to failure_history.json.
    Keeps last 50 entries per script.
    """
    history = load_failure_history()
    if script_name not in history:
        history[script_name] = []

    history[script_name].append({
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attempt"          : attempt,
        "error"            : error[:400],
        "exit_code"        : exit_code,
        "fix_applied"      : fix_applied,
        "layer_used"       : layer_used,      # L1 / L2 / L3
        "success_after_fix": success_after_fix,
    })

    # Cap history at 50 records per script
    history[script_name] = history[script_name][-50:]
    save_failure_history(history)


# =============================================================================
# LAYER 1 — SENTENCE TRANSFORMERS (Offline Semantic Matching)
# =============================================================================

def get_sentence_model():
    """Lazy-load Sentence Transformer model on first use."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            log("Loading Sentence Transformer model (first time only)...")
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            log("Sentence Transformer model loaded [OK]")
        except ImportError:
            log("sentence-transformers not installed. Run: pip install sentence-transformers", "WARNING")
            _sentence_model = None
    return _sentence_model


def build_knowledge_base() -> list:
    """
    Combine SEED_PATTERNS + past successful fixes from failure_history.json
    into a single knowledge base list for semantic matching.
    """
    kb = list(SEED_PATTERNS)   # start with seed patterns

    # Add past successful fixes from memory
    history = load_failure_history()
    for script_name, records in history.items():
        for record in records:
            if record.get("success_after_fix") and record.get("fix_applied") not in ("none", "log_only", ""):
                kb.append({
                    "error_text" : record["error"],
                    "fix_type"   : "past_experience",
                    "fix"        : record["fix_applied"],
                    "description": f"Past fix for {script_name} — worked on {record['timestamp']}",
                    "script_name": script_name,
                })

    return kb


def layer1_semantic_match(error_text: str) -> dict | None:
    """
    Layer 1: Use Sentence Transformers to find semantically similar
    errors in the knowledge base.
    Returns best matching fix or None if below threshold.
    """
    model = get_sentence_model()
    if model is None:
        return None   # fall through to Layer 2

    try:
        kb = build_knowledge_base()
        if not kb:
            return None

        # Encode the incoming error and all KB entries
        error_embedding = model.encode([error_text])
        kb_texts        = [entry["error_text"] for entry in kb]
        kb_embeddings   = model.encode(kb_texts)

        # Compute cosine similarities
        similarities = cosine_similarity(error_embedding, kb_embeddings)[0]
        best_idx     = int(np.argmax(similarities))
        best_score   = float(similarities[best_idx])

        log(f"  [L1] Best semantic match score: {best_score:.3f} | Entry: {kb[best_idx]['description']}")

        if best_score >= SIMILARITY_THRESHOLD:
            match = kb[best_idx]
            log(f"  [L1] MATCH accepted (score {best_score:.3f} >= {SIMILARITY_THRESHOLD})")
            return {
                "fix_type"   : match["fix_type"],
                "fix"        : match["fix"],
                "description": match["description"],
                "confidence" : round(best_score, 3),
                "layer"      : "L1_SentenceTransformer",
            }
        else:
            log(f"  [L1] NO MATCH (score {best_score:.3f} < {SIMILARITY_THRESHOLD})")
            return None

    except Exception as e:
        log(f"  [L1] Semantic matching error: {e}", "WARNING")
        return None


# =============================================================================
# LAYER 2 — GITHUB COPILOT API (AI Reasoning for Unknown Errors)
# =============================================================================

def layer2_copilot_analyse(error_text: str, script_name: str,
                            stdout: str, exit_code: int) -> dict | None:
    """
    Layer 2: Send error to GitHub Copilot API for intelligent analysis.
    Returns suggested fix or None if API unavailable.
    """
    if not GITHUB_TOKEN:
        log("  [L2] GITHUB_TOKEN not set — skipping Copilot analysis", "WARNING")
        return None

    # Build a rich prompt for Copilot
    prompt = f"""
You are a DevOps automation expert specialising in Windows batch scripts and DataStage ETL pipelines.

A batch script has failed. Analyse the error and respond ONLY with a JSON object.

Script Name : {script_name}
Exit Code   : {exit_code}
STDERR      : {error_text[:500]}
STDOUT      : {stdout[:300]}

Respond with ONLY this JSON (no explanation, no markdown):
{{
  "fix_type"   : "command | mkdir | wait | log_only",
  "fix"        : "<the actual fix — a shell command, directory path, wait seconds, or explanation>",
  "description": "<one line explaining the root cause>",
  "confidence" : <0.0 to 1.0>,
  "reasoning"  : "<brief reasoning for this fix>"
}}

fix_type rules:
- "command"  : a Windows shell command to run as a fix (e.g. icacls, net start, etc.)
- "mkdir"    : a directory path that needs to be created
- "wait"     : number of seconds to wait before retrying (e.g. "10")
- "log_only" : cannot be auto-fixed, human intervention needed
"""

    try:
        log(f"  [L2] Sending error to GitHub Copilot API for analysis...")

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type" : "application/json",
        }

        payload = {
            "model"      : COPILOT_MODEL,
            "temperature": 0.1,    # low temp for deterministic fix suggestions
            "messages"   : [
                {
                    "role"   : "system",
                    "content": "You are a DevOps expert. Always respond with valid JSON only."
                },
                {
                    "role"   : "user",
                    "content": prompt
                }
            ]
        }

        response = requests.post(
            COPILOT_API_URL,
            headers=headers,
            json=payload,
            timeout=COPILOT_TIMEOUT
        )

        if response.status_code != 200:
            log(f"  [L2] Copilot API error: HTTP {response.status_code} — {response.text[:200]}", "WARNING")
            return None

        raw_content = response.json()["choices"][0]["message"]["content"]

        # Clean JSON response (strip markdown fences if any)
        clean = raw_content.strip().replace("```json", "").replace("```", "").strip()
        suggestion = json.loads(clean)

        log(f"  [L2] Copilot suggested fix: [{suggestion.get('fix_type')}] {suggestion.get('description')}")
        log(f"  [L2] Reasoning: {suggestion.get('reasoning', 'N/A')}")

        return {
            "fix_type"   : suggestion.get("fix_type", "log_only"),
            "fix"        : suggestion.get("fix", ""),
            "description": suggestion.get("description", ""),
            "confidence" : suggestion.get("confidence", 0.5),
            "reasoning"  : suggestion.get("reasoning", ""),
            "layer"      : "L2_GitHubCopilot",
        }

    except json.JSONDecodeError as e:
        log(f"  [L2] Could not parse Copilot JSON response: {e}", "WARNING")
        return None
    except requests.Timeout:
        log(f"  [L2] Copilot API timed out after {COPILOT_TIMEOUT}s", "WARNING")
        return None
    except requests.ConnectionError:
        log(f"  [L2] Cannot reach Copilot API — no internet connection", "WARNING")
        return None
    except Exception as e:
        log(f"  [L2] Unexpected Copilot error: {e}", "WARNING")
        return None


# =============================================================================
# LAYER 3 — FAILURE HISTORY MEMORY (Past Experience Lookup)
# =============================================================================

def layer3_past_experience(script_name: str) -> dict | None:
    """
    Layer 3: Check failure_history.json for fixes that previously
    worked for this specific script.
    Returns the most recent successful fix or None.
    """
    history = load_failure_history()
    if script_name not in history:
        return None

    successful = [
        r for r in history[script_name]
        if r.get("success_after_fix")
        and r.get("fix_applied") not in ("none", "log_only", "")
    ]

    if not successful:
        return None

    # Return the most recently successful fix
    best = successful[-1]
    log(f"  [L3] PAST FIX found for '{script_name}': {best['fix_applied']} (worked on {best['timestamp']})")

    return {
        "fix_type"   : "past_experience",
        "fix"        : best["fix_applied"],
        "description": f"Re-applying fix that worked on {best['timestamp']}",
        "confidence" : 0.85,
        "layer"      : "L3_FailureMemory",
    }


# =============================================================================
# FIX APPLICATION ENGINE
# =============================================================================

def apply_fix(fix_type: str, fix: str, script_path: str) -> dict:
    """
    Execute the suggested fix based on its type.
    Returns success flag and message.
    """
    try:
        if fix_type in ("command", "past_experience"):
            # Resolve any placeholders
            cmd = (fix
                   .replace("{script_path}", script_path)
                   .replace("{output_dir}", OUTPUT_DIR))
            log(f"  Applying fix — running command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=15)
            return {
                "success": result.returncode == 0,
                "message": f"Ran: {cmd} | Exit: {result.returncode} | {result.stderr[:100]}"
            }

        elif fix_type == "mkdir":
            target = fix.replace("{output_dir}", OUTPUT_DIR)
            os.makedirs(target, exist_ok=True)
            log(f"  Applying fix — created directory: {target}")
            return {"success": True, "message": f"Created: {target}"}

        elif fix_type == "wait":
            secs = int(fix) if fix.isdigit() else RETRY_DELAY_SECS
            log(f"  Applying fix — waiting {secs} seconds")
            time.sleep(secs)
            return {"success": True, "message": f"Waited {secs}s"}

        elif fix_type == "log_only":
            log(f"  Fix is log_only — no auto-fix available: {fix}", "WARNING")
            return {"success": False, "message": fix}

        else:
            log(f"  Unknown fix_type: {fix_type}", "WARNING")
            return {"success": False, "message": f"Unknown fix_type: {fix_type}"}

    except Exception as e:
        return {"success": False, "message": f"Fix application error: {str(e)}"}


# =============================================================================
# CORE SELF-HEALING RUNNER
# =============================================================================

def run_script_with_healing(script_name: str) -> dict:
    """
    Runs a .bat script with full 3-layer self-healing:

    On failure:
      1. Check Layer 3 (past memory) for proactive fix
      2. Try running the script
      3. On failure -> Layer 1 (Sentence Transformers semantic match)
      4. If no L1 match -> Layer 2 (GitHub Copilot API analysis)
      5. Apply fix and retry
      6. Repeat up to MAX_RETRIES
      7. Escalate if still failing
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    # -- Pre-flight ------------------------------------------------------------
    if not os.path.exists(script_path):
        return {
            "success"  : False,
            "error"    : f"Script not found: {script_path}",
            "attempts" : 0,
            "healed"   : False,
            "layer"    : "none"
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    attempt     = 0
    last_error  = ""
    last_stdout = ""
    last_exit   = -1
    fix_applied = "none"
    layer_used  = "none"
    healed      = False

    # -- Proactive: Check Layer 3 memory BEFORE first run ---------------------
    proactive_fix = layer3_past_experience(script_name)
    if proactive_fix:
        log(f"  [L3 Proactive] Applying past successful fix before first attempt")
        apply_fix(proactive_fix["fix_type"], proactive_fix["fix"], script_path)
        fix_applied = proactive_fix["fix"]
        layer_used  = "L3_FailureMemory_Proactive"

    # -- Retry Loop ------------------------------------------------------------
    while attempt < MAX_RETRIES:
        attempt += 1
        log(f"  Attempt {attempt}/{MAX_RETRIES}: {script_name}")

        try:
            result = subprocess.run(
                script_path,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECS,
                shell=True
            )

            # -- SUCCESS -------------------------------------------------------
            if result.returncode == 0:
                log(f"  [SUCCESS] Attempt {attempt}: {script_name}")
                if attempt > 1:
                    healed = True
                    record_failure(script_name, last_error, last_exit,
                                   fix_applied, layer_used, attempt - 1, True)
                return {
                    "success"    : True,
                    "script"     : script_name,
                    "attempts"   : attempt,
                    "healed"     : healed,
                    "layer_used" : layer_used,
                    "fix_applied": fix_applied,
                    "stdout"     : result.stdout,
                    "stderr"     : result.stderr,
                }

            # -- FAILURE -------------------------------------------------------
            last_error  = result.stderr
            last_stdout = result.stdout
            last_exit   = result.returncode
            combined    = result.stderr + result.stdout

            log(f"  [FAILED] Attempt {attempt} failed (exit {result.returncode})", "WARNING")
            log(f"  stderr: {result.stderr[:200]}", "WARNING")

            if attempt >= MAX_RETRIES:
                break

            # -- Layer 1: Semantic Match ---------------------------------------
            log(f"  Consulting Layer 1 (Sentence Transformers)...")
            suggestion = layer1_semantic_match(combined)

            # -- Layer 2: GitHub Copilot fallback ------------------------------
            if suggestion is None:
                log(f"  Consulting Layer 2 (GitHub Copilot API)...")
                suggestion = layer2_copilot_analyse(
                    result.stderr, script_name, result.stdout, result.returncode
                )

            # -- Apply fix or fallback wait ------------------------------------
            if suggestion:
                fix_applied = suggestion["fix"]
                layer_used  = suggestion["layer"]
                log(f"  Applying [{layer_used}] fix: {suggestion['description']}")
                fix_result  = apply_fix(suggestion["fix_type"], suggestion["fix"], script_path)

                if not fix_result["success"] or suggestion["fix_type"] == "log_only":
                    log(f"  Fix cannot be applied automatically. Escalating.", "ERROR")
                    break
            else:
                # No fix from any layer — just wait and retry
                log(f"  No fix found from any layer — waiting {RETRY_DELAY_SECS}s and retrying")
                fix_applied = "unknown_wait"
                layer_used  = "none"
                time.sleep(RETRY_DELAY_SECS)

        except subprocess.TimeoutExpired:
            last_error  = f"Script timed out after {TIMEOUT_SECS}s"
            last_exit   = -1
            log(f"  [TIMEOUT] Attempt {attempt} timed out after {TIMEOUT_SECS}s", "WARNING")
            fix_applied = "timeout_wait"
            layer_used  = "timeout"
            time.sleep(RETRY_DELAY_SECS)

        except PermissionError:
            last_error = "Permission denied"
            last_exit  = -1
            log(f"  [PERMISSION DENIED] Attempt {attempt}", "ERROR")
            break

        except Exception as e:
            last_error = str(e)
            last_exit  = -1
            log(f"  [ERROR] Unexpected error: {e}", "ERROR")
            break

    # -- All retries exhausted -------------------------------------------------
    log(f"  [EXHAUSTED] All {attempt} attempts failed for: {script_name}", "ERROR")
    record_failure(script_name, last_error, last_exit,
                   fix_applied, layer_used, attempt, False)

    return {
        "success"    : False,
        "script"     : script_name,
        "attempts"   : attempt,
        "healed"     : False,
        "layer_used" : layer_used,
        "fix_applied": fix_applied,
        "error"      : last_error[:300],
        "exit_code"  : last_exit,
        "escalation" : f"'{script_name}' failed after {attempt} attempts. Manual intervention required.",
    }


# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config() -> dict:
    try:
        if not os.path.exists(CONFIG_FILE):
            return {"success": False, "error": f"Config not found: {CONFIG_FILE}"}
        with open(CONFIG_FILE, "r") as f:
            return {"success": True, "data": json.load(f)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in config: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def run_process(process_name: str) -> dict:
    """
    Runs a named process (e.g. 'daily A1', 'monthly B1', 'weekly C1').
    Looks up scripts in process_config.json, runs each in sequence.
    Automatically heals failures using 3-layer intelligence:
      Layer 1 - Sentence Transformers semantic match
      Layer 2 - GitHub Copilot AI analysis
      Layer 3 - Past failure memory
    """
    log(f"START process='{process_name}'")

    try:
        config_result = load_config()
        if not config_result["success"]:
            return {"success": False, "error": config_result["error"], "results": []}

        processes = config_result["data"].get("processes", {})

        # Fuzzy match process name from prompt
        matched_key = None
        for key in processes.keys():
            if key.lower() in process_name.lower().strip():
                matched_key = key
                break

        if not matched_key:
            return {
                "success"            : False,
                "error"              : f"Process '{process_name}' not found",
                "available_processes": list(processes.keys()),
                "results"            : []
            }

        scripts      = processes[matched_key]["scripts"]
        healed_count = 0
        results      = []

        log(f"FOUND '{matched_key}' — {len(scripts)} script(s)")

        for i, script in enumerate(scripts, 1):
            log(f"Script {i}/{len(scripts)}: {script}")
            result = run_script_with_healing(script)
            results.append(result)

            if result.get("healed"):
                healed_count += 1

            if not result["success"]:
                log(f"STOPPED at script {i} — escalating", "ERROR")
                return {
                    "success"      : False,
                    "process"      : matched_key,
                    "failed_script": script,
                    "failed_at"    : f"{i}/{len(scripts)}",
                    "healed_count" : healed_count,
                    "results"      : results,
                    "escalation"   : result.get("escalation"),
                }

        log(f"END SUCCESS '{matched_key}' | healed={healed_count}")
        return {
            "success"      : True,
            "process"      : matched_key,
            "total_scripts": len(scripts),
            "healed_count" : healed_count,
            "results"      : results,
        }

    except Exception as e:
        log(f"UNEXPECTED ERROR: {e}", "ERROR")
        return {"success": False, "error": str(e),
                "stderr": traceback.format_exc(), "results": []}


@mcp.tool()
def get_failure_history(script_name: str = "") -> dict:
    """
    Returns past failure history with fix outcomes.
    Leave script_name blank to get all history.
    Example: 'Show failure history for daily_process1.bat'
    Example: 'Show all failure history'
    """
    try:
        history = load_failure_history()
        if not history:
            return {"success": True, "message": "No failure history yet", "history": {}}
        if script_name:
            return {"success": True, "script": script_name,
                    "history": history.get(script_name, [])}
        return {"success": True, "total_scripts": len(history), "history": history}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def clear_failure_history(script_name: str = "") -> dict:
    """
    Clears failure history memory.
    Pass script name to clear one, or leave blank to clear all.
    Example: 'Clear failure history for daily_process1.bat'
    Example: 'Clear all failure history'
    """
    try:
        if script_name:
            history = load_failure_history()
            if script_name in history:
                del history[script_name]
                save_failure_history(history)
                return {"success": True, "message": f"Cleared: {script_name}"}
            return {"success": True, "message": f"No history found for: {script_name}"}
        save_failure_history({})
        return {"success": True, "message": "All failure history cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_processes() -> dict:
    """
    Lists all available processes and their scripts.
    Example: 'List all available processes'
    """
    try:
        config_result = load_config()
        if not config_result["success"]:
            return {"success": False, "error": config_result["error"]}
        processes = config_result["data"].get("processes", {})
        return {
            "success"  : True,
            "total"    : len(processes),
            "processes": {
                k: {
                    "description" : v.get("description", ""),
                    "script_count": len(v["scripts"]),
                    "scripts"     : v["scripts"]
                }
                for k, v in processes.items()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()