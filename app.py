import gradio as gr
from pipeline import MultimodalPipeline

# Initialize the pipeline
# Note: It handles missing models gracefully with dummy predictions for testing
pipeline = MultimodalPipeline()

def analyze_news(text, image):
    """Gradio interface function."""
    if not text.strip() and image is None:
        return (
            "Please provide text, an image, or both.",
            gr.update(value=0, visible=False),
            gr.update(value=0, visible=False),
            gr.update(value=0, visible=False),
            "No inputs provided."
        )
    
    result = pipeline.predict(text, image)
    if not result:
        return "Error in processing.", gr.update(), gr.update(), gr.update(), "Error"
    
    verdict = result["verdict"]
    text_score = result.get("text_score")
    image_score = result.get("image_score")
    combined_score = result["combined_score"]
    logic = result["logic"]

    # Formatting output elements
    verdict_html = f"<h2 style='text-align:center;'>Verdict: {verdict}</h2>"
    if verdict == "High Risk / Fake News":
        verdict_html = f"<h2 style='text-align:center; color:#FF4C4C;'>Verdict: {verdict} 🚨</h2>"
    elif verdict == "Suspicious / Misleading":
        verdict_html = f"<h2 style='text-align:center; color:#FFA500;'>Verdict: {verdict} ⚠️</h2>"
    else:
        verdict_html = f"<h2 style='text-align:center; color:#4CAF50;'>Verdict: {verdict} ✅</h2>"

    explanation = f"**Logic Used**: {logic}<br/>"
    if text_score is not None and image_score is not None:
        explanation += "Both modalities were analyzed. The final score is based on the maximum risk detected across text and image."
    elif text_score is not None:
        explanation += "Only text was analyzed."
    else:
        explanation += "Only image was analyzed."

    # Convert scores to percentages for progress bars
    t_val = text_score if text_score is not None else 0.0
    i_val = image_score if image_score is not None else 0.0
    
    t_visible = text_score is not None
    i_visible = image_score is not None

    return (
        verdict_html,
        gr.update(value=t_val, visible=t_visible),
        gr.update(value=i_val, visible=i_visible),
        gr.update(value=combined_score, visible=True),
        explanation
    )


# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    
    gr.Markdown(
        """
        # 🕵️‍♂️ Multimodal Fake News Detector
        Upload a news headline/article snippet and its accompanying image to check its authenticity.
        Our AI analyzes both **Textual Patterns** and **Visual Artifacts** to determine a Multimodal Trust Score.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Modalities")
            text_input = gr.Textbox(
                lines=5, 
                placeholder="Enter news text, headline, or article snippet here...",
                label="News Text"
            )
            image_input = gr.Image(type="pil", label="News Image")
            
            analyze_btn = gr.Button("Analyze News 🔍", variant="primary")
            
            gr.Examples(
                examples=[
                    ["Shocking: Alien spacecraft found in Antarctica!", None],
                    ["Federal Reserve announces a 0.25% interest rate hike.", None],
                    ["You won't believe what this celebrity did next! Revealed!", None]
                    # Note: You can add paths to example images if you place them in an 'examples' folder
                ],
                inputs=[text_input, image_input]
            )

        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")
            
            verdict_output = gr.HTML(value="<h2 style='text-align:center; color:gray;'>Awaiting Input...</h2>")
            
            gr.Markdown("#### Confidence Breakdown (Fake Probability)")
            text_prob = gr.Slider(minimum=0, maximum=1, label="Text Fake Probability", interactive=False, visible=False)
            image_prob = gr.Slider(minimum=0, maximum=1, label="Image Fake Probability", interactive=False, visible=False)
            combined_prob = gr.Slider(minimum=0, maximum=1, label="Combined Multimodal Risk Score", interactive=False, visible=False)
            
            explanation_output = gr.Markdown("Submit an article to see detailed feedback.")

    analyze_btn.click(
        fn=analyze_news,
        inputs=[text_input, image_input],
        outputs=[verdict_output, text_prob, image_prob, combined_prob, explanation_output]
    )

if __name__ == "__main__":
    demo.launch(share=False) # Set share=True for a public URL
