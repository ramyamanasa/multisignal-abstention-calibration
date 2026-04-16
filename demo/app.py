"""
Demo: Gradio UI
Owner: Person C (Integration + Demo)
"""
import gradio as gr

def predict(question: str, threshold: float):
    # TODO: replace with real pipeline call
    return (
        "Stub answer",       # answer
        0.3,                  # mean_entropy
        0.4,                  # semantic_inconsistency
        0.2,                  # cross_model_disagreement
        0.25,                 # fused hallucination probability
        "ANSWER"              # decision
    )

with gr.Blocks(title="Multi-Signal Hallucination Detector") as demo:
    gr.Markdown("# Multi-Signal Uncertainty Fusion")
    gr.Markdown("Enter a factual question. The system computes hallucination risk and decides whether to answer or abstain.")

    with gr.Row():
        question_input = gr.Textbox(label="Question", placeholder="e.g. Who wrote Hamlet?")
        threshold_slider = gr.Slider(0.3, 0.95, value=0.5, step=0.05, label="Abstention Threshold (higher = answer more)")

    submit_btn = gr.Button("Run")

    with gr.Row():
        answer_out = gr.Textbox(label="Answer")
        decision_out = gr.Textbox(label="Decision")

    with gr.Row():
        entropy_out = gr.Number(label="Token Entropy")
        consistency_out = gr.Number(label="Semantic Inconsistency")
        disagreement_out = gr.Number(label="Cross-Model Disagreement")
        prob_out = gr.Number(label="P(Hallucination)")

    submit_btn.click(
        fn=predict,
        inputs=[question_input, threshold_slider],
        outputs=[answer_out, entropy_out, consistency_out, disagreement_out, prob_out, decision_out]
    )

if __name__ == "__main__":
    demo.launch()
