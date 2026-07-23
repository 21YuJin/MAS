"""
[Step 4-2] OllamaAgentExecutor -- the ONE entry point every real Ollama call
in the travel_a2a Ollama-backed agents goes through. No agent implementation
(ollama_agents.py) calls runtime.ollama_client.ask_ollama directly or
hand-copies prompt_eval_count/eval_count/durations/done_reason itself --
every one of those fields is captured here, once, into an AgentCallRecord
(Step 4-1), so a future new Ollama agent can never accidentally forget a
telemetry field or duplicate the capture logic slightly differently.

execute() does exactly what the Step 4-2 instruction lists:
  1. call_start_timestamp
  2. (prompt is built by the caller -- OllamaAgentExecutor does not decide
     prompt content, see prompt_builders.py)
  3. runtime.ollama_client.ask_ollama()
  4. collect Ollama raw telemetry (the dict ask_ollama already returns)
  5. call_end_timestamp
  6. build the AgentCallRecord
  7. build the response Part
  8. (Artifact creation, if any, is the caller's job -- see dispatch.py's
     create_artifact() -- since not every LLM call produces an Artifact)
  9. retry/error bookkeeping (passthrough from ask_ollama today; ask_ollama
     itself has no retry loop yet -- see its own docstring)
"""
import dataclasses
import datetime as dt
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runtime.ollama_client import ask_ollama  # noqa: E402  [Step 1-1] shared client

from .models import AgentCallRecord, Part, PartType, SourceType

DEFAULT_MODEL = "llama3.2"


@dataclasses.dataclass
class AgentExecutionResult:
    response_text: str
    response_part: Part
    call_record: AgentCallRecord
    parse_error: Optional[str] = None


class OllamaAgentExecutor:
    def __init__(self, model: str = DEFAULT_MODEL, prompt_config_version: str = "v1",
                 agent_config_version: str = "v1"):
        self.model = model
        self.prompt_config_version = prompt_config_version
        self.agent_config_version = agent_config_version

    def execute(self, agent_id: str, action_type: str, prompt: str, session_id: str, task_id: str,
                context_id: str, id_factory, triggering_message_id: Optional[str] = None,
                input_part_ids: Optional[List[str]] = None, input_artifact_ids: Optional[List[str]] = None,
                seed: Optional[int] = None) -> AgentExecutionResult:
        call_start = dt.datetime.now(dt.timezone.utc).isoformat()
        raw = ask_ollama(prompt, seed=seed, model=self.model)
        call_end = dt.datetime.now(dt.timezone.utc).isoformat()

        response_part = Part(
            part_id=id_factory.part_id(), part_type=PartType.TEXT, mime_type="text/plain",
            content=raw["text"], source_type=SourceType.AGENT_GENERATED,
            injection_present=False, attack_id=None, created_at=call_end,
        )

        call_record = AgentCallRecord(
            call_id=id_factory.call_id(), session_id=session_id, task_id=task_id, context_id=context_id,
            agent_id=agent_id, action_type=action_type, triggering_message_id=triggering_message_id,
            input_part_ids=(input_part_ids or []), input_artifact_ids=(input_artifact_ids or []),
            output_part_ids=[response_part.part_id], output_artifact_ids=[],
            call_start_timestamp=call_start, call_end_timestamp=call_end,
            wall_clock_latency_ms=raw["wall_clock_latency_ms"],
            llm_called=True, model_name=raw["model"], model_digest=None,
            prompt_eval_count=raw["prompt_eval_count"], eval_count=raw["eval_count"],
            prompt_eval_duration=raw["prompt_eval_duration"], eval_duration=raw["eval_duration"],
            total_duration=raw["total_duration"], load_duration=raw["load_duration"],
            done_reason=raw["done_reason"], retry_count=raw["retry_count"], error_flag=raw["error_flag"],
            error_type=(None if not raw["error_flag"] else "ollama_request_error"),
            temperature=raw["temperature"], top_p=raw["top_p"], seed=seed,
            prompt_config_version=self.prompt_config_version, agent_config_version=self.agent_config_version,
            raw_ollama_telemetry=raw, timing_source="ollama_runtime",
        )
        return AgentExecutionResult(response_text=raw["text"], response_part=response_part, call_record=call_record)
