"""
News Fact-Checker Demo
Multi-Signal Hallucination Abstention · STAT GR5293 | Spring 2026
"""

import json
import difflib
import gradio as gr

CACHE_PATH = "../data/processed/news_demo_cache.json"
DEFAULT_THRESHOLD = 0.86

with open(CACHE_PATH) as f:
    _cache = json.load(f)

QUESTIONS = _cache["questions"]
DOMAINS   = ["Breaking News", "Science", "History"]
BY_DOMAIN = {d: [q for q in QUESTIONS if q["domain"] == d] for d in DOMAINS}
BY_Q      = {q["question"]: q for q in QUESTIONS}
ALL_Q_TEXT = list(BY_Q.keys())

SIGNAL_META = {
    "mean_entropy":             ("Mean Token Entropy",       "Average uncertainty across all generated tokens"),
    "max_entropy":              ("Max Token Entropy",         "Uncertainty at the single most uncertain word"),
    "entity_entropy":           ("Entity Entropy",            "Uncertainty specifically at named entities (names, dates, places)"),
    "semantic_inconsistency":   ("Semantic Inconsistency",    "Spread across 5 independent Groq samples — high = model keeps changing its answer"),
    "cross_model_disagreement": ("Cross-Model Disagreement",  "Embedding distance between small (opt-125m) and large (llama-3.1-8b) model answers"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fuzzy_match(text: str) -> str | None:
    text = text.strip()
    hits = difflib.get_close_matches(text, ALL_Q_TEXT, n=1, cutoff=0.3)
    if hits:
        return hits[0]
    low = text.lower()
    for q in ALL_Q_TEXT:
        if low in q.lower() or q.lower() in low:
            return q
    return None


def signal_bars_html(signals: dict) -> str:
    rows = ""
    for key, (label, tip) in SIGNAL_META.items():
        val = signals.get(key)
        if val is None:
            color, width = "#9ca3af", 0
        elif val < 0.3:
            color, width = "#22c55e", val
        elif val < 0.6:
            color, width = "#f59e0b", val
        else:
            color, width = "#ef4444", val
        pct   = min(100, max(0, width * 100))
        v_str = f"{val:.3f}" if val is not None else "—"
        rows += f"""
        <div style="margin:9px 0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="font-size:12px;color:#374151;font-weight:500;">{label}</span>
            <span style="font-size:12px;font-weight:700;color:{color};">{v_str}</span>
          </div>
          <div style="background:#e5e7eb;border-radius:6px;height:11px;overflow:hidden;">
            <div style="background:{color};width:{pct:.1f}%;height:11px;border-radius:6px;"></div>
          </div>
          <div style="font-size:10px;color:#9ca3af;margin-top:2px;">{tip}</div>
        </div>"""
    return f"<div style='font-family:system-ui,sans-serif;padding:4px 0;'>{rows}</div>"


def why_html(c: dict, decision: str, threshold: float) -> str:
    s = c["signals"]
    if decision == "ABSTAIN":
        parts = []
        if s.get("semantic_inconsistency") is not None and s["semantic_inconsistency"] > 0.25:
            parts.append(f"the model gave varying answers across 5 independent samples "
                         f"(inconsistency = <b>{s['semantic_inconsistency']:.2f}</b>)")
        if s.get("cross_model_disagreement") is not None and s["cross_model_disagreement"] > 0.5:
            parts.append(f"a small and a large AI model strongly disagreed "
                         f"(disagreement = <b>{s['cross_model_disagreement']:.2f}</b>)")
        if s.get("max_entropy") is not None and s["max_entropy"] > 0.3:
            parts.append(f"the model was uncertain about key words "
                         f"(max entropy = <b>{s['max_entropy']:.2f}</b>)")
        if not parts:
            parts.append(f"the fused hallucination score (<b>{c['hallucination_prob']:.2f}</b>) "
                         f"exceeds the threshold (<b>{threshold:.2f}</b>)")
        return "🚨 <b>Abstaining because</b> " + ", and ".join(parts) + "."
    else:
        parts = []
        if s.get("semantic_inconsistency") is not None and s["semantic_inconsistency"] < 0.2:
            parts.append(f"the model gave consistent answers across samples "
                         f"(inconsistency = <b>{s['semantic_inconsistency']:.2f}</b>)")
        if s.get("cross_model_disagreement") is not None and s["cross_model_disagreement"] < 0.4:
            parts.append(f"both AI models agreed on this answer "
                         f"(disagreement = <b>{s['cross_model_disagreement']:.2f}</b>)")
        if s.get("max_entropy") is not None and s["max_entropy"] < 0.2:
            parts.append(f"the model was confident in its word choices "
                         f"(max entropy = <b>{s['max_entropy']:.2f}</b>)")
        if not parts:
            parts.append(f"the hallucination score (<b>{c['hallucination_prob']:.2f}</b>) "
                         f"is below the threshold (<b>{threshold:.2f}</b>)")
        return "✅ <b>Answering because</b> " + ", and ".join(parts) + "."


def coverage_html(threshold: float) -> str:
    n   = len(QUESTIONS)
    answered = sum(1 for q in QUESTIONS if q["hallucination_prob"] < threshold)
    correct  = sum(
        1 for q in QUESTIONS
        if (q["hallucination_prob"] < threshold) == (q["expected_decision"] == "ANSWER")
    )
    cov = answered / n
    acc = correct / n

    rows = ""
    for dom in DOMAINS:
        dqs   = BY_DOMAIN[dom]
        d_ans = sum(1 for q in dqs if q["hallucination_prob"] < threshold)
        d_abs = len(dqs) - d_ans
        rows += (
            f"<tr>"
            f"<td style='padding:4px 10px;color:#374151;font-size:12px;'>{dom}</td>"
            f"<td style='padding:4px 10px;text-align:center;color:#16a34a;font-weight:600;font-size:12px;'>{d_ans}</td>"
            f"<td style='padding:4px 10px;text-align:center;color:#dc2626;font-weight:600;font-size:12px;'>{d_abs}</td>"
            f"</tr>"
        )

    return f"""
    <div style="font-family:system-ui,sans-serif;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;padding:16px;">
      <h4 style="margin:0 0 12px 0;color:#1e293b;font-size:14px;">
        📊 Coverage–Accuracy Tracker
        <span style="font-weight:400;color:#94a3b8;font-size:12px;"> threshold = {threshold:.2f}</span>
      </h4>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:white;border-radius:8px;padding:10px;text-align:center;border:1px solid #dbeafe;">
          <div style="font-size:28px;font-weight:800;color:#2563eb;">{cov:.0%}</div>
          <div style="font-size:11px;color:#64748b;margin-top:2px;">Coverage<br>{answered}/{n} questions answered</div>
        </div>
        <div style="background:white;border-radius:8px;padding:10px;text-align:center;border:1px solid #ede9fe;">
          <div style="font-size:28px;font-weight:800;color:#7c3aed;">{acc:.0%}</div>
          <div style="font-size:11px;color:#64748b;margin-top:2px;">Policy Stability<br>decisions unchanged from 0.86</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;border:1px solid #f1f5f9;">
        <tr style="background:#f8fafc;">
          <th style="padding:5px 10px;text-align:left;font-size:11px;color:#64748b;font-weight:600;">Domain</th>
          <th style="padding:5px 10px;font-size:11px;color:#16a34a;font-weight:600;">ANSWER</th>
          <th style="padding:5px 10px;font-size:11px;color:#dc2626;font-weight:600;">ABSTAIN</th>
        </tr>
        {rows}
      </table>
    </div>"""


def decision_card_html(c: dict, threshold: float, matched_note: str = "") -> str:
    decision = "ABSTAIN" if c["hallucination_prob"] >= threshold else "ANSWER"
    prob     = c["hallucination_prob"]

    if decision == "ANSWER":
        bg, border, badge, icon = "#f0fdf4", "#22c55e", "#16a34a", "✅"
    else:
        bg, border, badge, icon = "#fef2f2", "#ef4444", "#dc2626", "🚫"

    note_html = (
        f"<div style='font-size:11px;color:#94a3b8;margin-bottom:6px;'>"
        f"Matched to: <em>{matched_note}</em></div>"
        if matched_note else ""
    )

    return f"""
    <div style="padding:22px;background:{bg};border-radius:16px;border:3px solid {border};font-family:system-ui,sans-serif;">
      {note_html}
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;">
        <span style="font-size:38px;line-height:1;">{icon}</span>
        <span style="font-size:32px;font-weight:900;color:{badge};letter-spacing:-0.5px;">{decision}</span>
        <span style="background:{badge};color:white;padding:5px 14px;border-radius:999px;font-size:13px;font-weight:600;">
          P(hallucination) = {prob:.3f}
        </span>
      </div>
      <div style="font-size:14px;color:#111827;margin-bottom:6px;">
        <span style="color:#6b7280;font-weight:500;">Q:</span> {c['question']}
      </div>
      <div style="font-size:14px;color:#111827;margin-bottom:12px;">
        <span style="color:#6b7280;font-weight:500;">A:</span>
        <strong>{c.get('groq_answer', '—')}</strong>
      </div>
      <div style="font-size:13px;color:#374151;background:rgba(0,0,0,0.04);padding:10px 14px;border-radius:8px;line-height:1.6;">
        {why_html(c, decision, threshold)}
      </div>
    </div>"""


EMPTY_CARD = ("<div style='padding:24px;color:#9ca3af;font-style:italic;font-family:system-ui,sans-serif;"
              "border:2px dashed #e5e7eb;border-radius:16px;text-align:center;'>"
              "Select a question from a tab above, or type one below.</div>")


def run_check(question_text: str, threshold: float):
    cov = coverage_html(threshold)
    if not question_text or not question_text.strip():
        return EMPTY_CARD, "", cov

    matched_note = ""
    if question_text in BY_Q:
        c = BY_Q[question_text]
    else:
        hit = fuzzy_match(question_text)
        if hit:
            c = BY_Q[hit]
            if hit != question_text:
                matched_note = hit
        else:
            return (
                "<div style='padding:20px;background:#fef9c3;border-radius:12px;border:2px solid #facc15;"
                "font-family:system-ui,sans-serif;'><b>❓ Not in cache</b> — No close match found. "
                "Try a preset question from the tabs above.</div>",
                "",
                cov,
            )

    return (
        decision_card_html(c, threshold, matched_note),
        signal_bars_html(c["signals"]),
        cov,
    )


# ── Build UI ──────────────────────────────────────────────────────────────────

HEADER_MD = """
# 📰 News Fact-Checker
### Multi-Signal Hallucination Abstention · STAT GR5293 | Spring 2026

Three independent uncertainty signals — **token entropy**, **semantic consistency**, and **cross-model disagreement** — are fused into a calibrated hallucination probability.
The system **answers** when confident and **abstains** when not.
Move the threshold slider to see the coverage/accuracy trade-off in real time.
"""

with gr.Blocks(title="News Fact-Checker") as demo:

    gr.Markdown(HEADER_MD)

    with gr.Row(equal_height=False):

        # ── Left column: controls ─────────────────────────────────────────────
        with gr.Column(scale=2, min_width=320):

            with gr.Tabs():
                for dom in DOMAINS:
                    with gr.Tab(dom):
                        choices = [q["question"] for q in BY_DOMAIN[dom]]
                        dd = gr.Dropdown(
                            choices=choices,
                            label=f"Preset {dom} questions",
                            value=None,
                            elem_id=f"dd_{dom.replace(' ', '_')}",
                        )
                        # Each dropdown click populates question box and runs check
                        # (wired below after all components are defined)
                        setattr(demo, f"_dd_{dom}", dd)

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Or type any question here — fuzzy-matched to cache…",
                    label="Question",
                    lines=2,
                    scale=4,
                )
                check_btn = gr.Button("Check ↵", variant="primary", scale=1, min_width=80)

            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                label="Abstention Threshold",
                info="ABSTAIN if P(hallucination) ≥ threshold.  Higher → answer more.  Lower → abstain more.",
            )

            gr.Markdown("""
**How the three signals work:**
- **Token Entropy** — measured on `facebook/opt-125m`. High entropy at named-entity positions flags factual uncertainty.
- **Semantic Consistency** — 5 stochastic Groq samples. High spread = the model itself isn't sure.
- **Cross-Model Disagreement** — embedding distance between opt-125m and llama-3.1-8b answers. Large gap = hallucination risk.
- **Fusion** — calibrated logistic regression with isotonic regression (AUROC 0.88 on held-out HaluEval).
""")

        # ── Right column: results ─────────────────────────────────────────────
        with gr.Column(scale=3, min_width=400):

            decision_out = gr.HTML(value=EMPTY_CARD)

            with gr.Accordion("Signal Decomposition", open=True):
                signals_out = gr.HTML(
                    value="<div style='color:#9ca3af;font-style:italic;font-size:13px;'>"
                          "Signal bars appear after you run a question.</div>"
                )

            coverage_out = gr.HTML(value=coverage_html(DEFAULT_THRESHOLD))

    # ── Event wiring ──────────────────────────────────────────────────────────

    outputs = [decision_out, signals_out, coverage_out]

    for dom in DOMAINS:
        dd = getattr(demo, f"_dd_{dom}")

        # Populate text box when dropdown is used
        dd.change(fn=lambda q: q or "", inputs=[dd], outputs=[question_box])

        # Run check immediately
        dd.change(
            fn=lambda q, t: run_check(q, t) if q else (EMPTY_CARD, "", coverage_html(t)),
            inputs=[dd, threshold_slider],
            outputs=outputs,
        )

    check_btn.click(fn=run_check, inputs=[question_box, threshold_slider], outputs=outputs)
    question_box.submit(fn=run_check, inputs=[question_box, threshold_slider], outputs=outputs)

    # Threshold slider always re-runs if there is a question, and always updates coverage
    threshold_slider.change(
        fn=lambda q, t: run_check(q, t) if q and q.strip() else (EMPTY_CARD, "", coverage_html(t)),
        inputs=[question_box, threshold_slider],
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
