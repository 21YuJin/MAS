"""
[Step 1-1, shared runtime foundation] Single canonical `ask_ollama()` --
previously duplicated three times (lgnn_experiment.py, collect_normal.py in
full, mini_validation.py in a REDUCED form missing prompt_eval_duration/
eval_duration/total_duration/load_duration/retry_count/end_timestamp/model/
temperature/top_p/num_predict). All three callers now import this module
instead, so a session collected by any of the three scripts is telemetry-
schema-identical to the others.

Field names are unchanged from the original three implementations (pure
extraction, no renames) so every existing call site keeps working with zero
call-signature changes -- this IS the "compatibility wrapper": callers that
only ever read a subset of the returned dict (e.g. mini_validation.py reading
just text/eval_count) are unaffected by the extra fields now present.

Two fields are new here (not present in any of the three original
implementations):
  - created_at:   Ollama's OWN reported response timestamp (data["created_at"]),
                   distinct from this function's start_timestamp/end_timestamp
                   (our wrapper's wall-clock measurements around the call).
  - raw_response: the full parsed JSON body, so a future feature that needs a
                   field nobody extracted yet can read it from already-collected
                   data instead of forcing recollection (same "raw first,
                   derive features later" principle as the rest of this dict).
"""
import datetime as dt
import time

import requests

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def ask_ollama(prompt, seed=None, model=DEFAULT_MODEL, url=DEFAULT_URL, timeout=120):
    """
    Returns a RAW telemetry dict -- everything Ollama's /api/generate reports
    plus our own wrapper metadata. error_flag=True / ok=False on request
    exception or an empty response; retry_count is always 0 (no retry loop
    implemented). The eval_count=30 fallback on exception is a pre-existing
    behavior carried over unchanged from every prior implementation of this
    function -- not a new default introduced here.
    """
    start_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    start = time.time()
    options = {}
    if seed is not None:
        options["seed"] = seed
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        data = r.json()
        text = data.get("response", "")
        wall_clock_latency_ms = round((time.time() - start) * 1000, 2)
        end_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        return {
            "text": text,
            "ok": bool(text),
            "error_flag": False,
            "retry_count": 0,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count", len(text.split())),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "wall_clock_latency_ms": wall_clock_latency_ms,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "created_at": data.get("created_at"),
            "model": data.get("model", model),
            "temperature": options.get("temperature"),
            "top_p": options.get("top_p"),
            "num_predict": options.get("num_predict"),
            "done_reason": data.get("done_reason"),
            "raw_response": data,
        }
    except Exception:
        wall_clock_latency_ms = round((time.time() - start) * 1000, 2)
        end_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        return {
            "text": "", "ok": False, "error_flag": True, "retry_count": 0,
            "prompt_eval_count": None, "eval_count": 30,
            "prompt_eval_duration": None, "eval_duration": None,
            "total_duration": None, "load_duration": None,
            "wall_clock_latency_ms": wall_clock_latency_ms,
            "start_timestamp": start_timestamp, "end_timestamp": end_timestamp,
            "created_at": None,
            "model": model, "temperature": options.get("temperature"),
            "top_p": options.get("top_p"), "num_predict": options.get("num_predict"),
            "done_reason": None,
            "raw_response": None,
        }
