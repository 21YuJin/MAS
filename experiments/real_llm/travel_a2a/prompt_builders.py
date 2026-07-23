"""
[Step 4-3] Prompt construction for the Ollama-backed travel agents --
enforces the prompt/metadata boundary: a prompt may only ever contain the
agent's role instruction (prompts/*.txt), the relevant slice of the
TravelRequest, and already-retrieved external content (option
descriptions/provider notes -- the same "content a real specialist would
actually read" a human travel agent would see). A prompt must NEVER contain
any of FORBIDDEN_PROMPT_FIELDS below -- in particular, a task fixture's
expected_branches is test-only ground truth and must never reach the LLM.

Each build_*_prompt() function returns (prompt_text, structured_data) --
structured_data is the SAME dict content_repository already returned (never
reinterpreted from the LLM's own output), so workflow_policy.py continues to
read only structured, deterministic values for branch decisions (per the
Step 3 feedback: policy must never decide a branch by reading raw text an
LLM produced). The LLM's response becomes an ADDITIONAL narrative Part
attached alongside the structured one -- never a replacement for it.
"""
import os

FORBIDDEN_PROMPT_FIELDS = {
    "injection_present", "attack_id", "attack_goal", "condition", "ground_truth_label",
    "expected_branches", "expected_effect", "evaluator_criterion", "indicator_observed",
    "goal_success", "propagation_observed",
}

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read()


def _render_options(options: list, price_key: str) -> str:
    lines = []
    for opt in options:
        lines.append(f"- {opt['option_id']}: {opt.get('description', '')} "
                      f"({opt[price_key]} {opt['currency']})")
    return "\n".join(lines) if lines else "(no options available)"


def build_flight_prompt(destination: str, options: list) -> str:
    template = _load_template("flight_v1.txt")
    rendered = _render_options(options, "price")
    return f"{template}\nDestination: {destination}\nFlight options:\n{rendered}\n"


def build_hotel_prompt(destination: str, options: list) -> str:
    template = _load_template("hotel_v1.txt")
    rendered = _render_options(options, "total_price")
    return f"{template}\nDestination: {destination}\nHotel options:\n{rendered}\n"


def build_currency_prompt(conversion: dict) -> str:
    template = _load_template("currency_v1.txt")
    body = (f"Budget: {conversion['total_budget_target_currency']} {conversion['target_currency']} "
            f"(converted from {conversion['base_currency']} at rate {conversion['rate']}).\n"
            f"Lodging budget allocation: {conversion['lodging_budget_target_currency']} {conversion['target_currency']}.")
    return f"{template}\n{body}\n"


def build_tours_prompt(destination: str, options: list) -> str:
    template = _load_template("tours_v1.txt")
    rendered = _render_options(options, "price")
    return f"{template}\nDestination: {destination}\nTour/activity options:\n{rendered}\n"


def build_coordinator_prompt(destination: str, artifact_summary: dict) -> str:
    template = _load_template("coordinator_v1.txt")
    lines = [f"- {atype}: version {info['version']}, {info['record_count']} record(s)"
              for atype, info in artifact_summary.items()]
    return f"{template}\nDestination: {destination}\nGathered results:\n" + "\n".join(lines) + "\n"
