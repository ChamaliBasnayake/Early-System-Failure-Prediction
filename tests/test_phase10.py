"""Phase 10 — Gemini evaluation plumbing (no API calls)."""
import re
from pathlib import Path

from src.genai.gemini_eval import PROMPT_DIR, parse_prob

PROVIDED = {"N", "N_plus_H", "max_lines", "examples", "local", "global", "n_local", "k_local", "n_unk_local", "n_global",
            "n_units_active_global", "n_nontarget_alerts_global", "has_nontarget_alert_local", "hour", "dow"}


def test_parse_prob_json_and_fallback():
    assert parse_prob('{"probability": 0.72, "reason": "burst"}') == (0.72, "burst")
    assert parse_prob("probability: 1.4")[0] == 1.0          # clipped
    assert parse_prob("no number")[0] != parse_prob("no number")[0]   # NaN


def test_prompts_have_only_known_placeholders():
    files = sorted(PROMPT_DIR.glob("p*.txt")); assert len(files) == 4
    for f in files:
        ph = set(re.findall(r"(?<!\{)\{([a-zA-Z_]+)\}(?!\})", f.read_text(encoding="utf-8")))
        assert ph <= PROVIDED, (f.name, ph - PROVIDED)
        assert '{{"probability"' in f.read_text(encoding="utf-8")   # JSON answer format requested
