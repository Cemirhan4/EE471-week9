"""
EE471 Week 9 — Task 3: Interactive Gradio Demo
Provides a 10-tab web front-end powered by the OOP HuggingFaceTasks module.
"""

import gradio as gr
from hf_tasks import HuggingFaceTasks


tasks = HuggingFaceTasks()


with gr.Blocks(title="EE471 Week 9 — HuggingFace AI Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# EE471 Week #9 — HuggingFace Transformers Interactive Demo\n"
        "OOP-powered Gradio suite covering all 10 NLP, Vision, and Audio tasks."
    )

    with gr.Tabs():
        # 1. Sentiment Analysis
        with gr.Tab("1. Sentiment Analysis"):
            sa_in = gr.Textbox(
                label="Input Sentences (one per line)",
                lines=3,
                value="I've been waiting for a modern software practices course my whole life.\nI hate software courses",
            )
            sa_out = gr.Label(label="Sentiment Predictions")
            gr.Button("Analyze Sentiment", variant="primary").click(
                tasks.sentiment_analysis, inputs=sa_in, outputs=sa_out
            )

        # 2. Zero-Shot Classification
        with gr.Tab("2. Zero-Shot Classification"):
            zs_in = gr.Textbox(
                label="Sequence",
                value="Berkshire keeps their cash reserves at an extremely high level.",
            )
            zs_labels = gr.Textbox(
                label="Candidate Labels (comma-separated)",
                value="finance, business, politics, education, sports",
            )
            zs_out = gr.Label(label="Zero-Shot Probabilities")
            gr.Button("Classify", variant="primary").click(
                tasks.zero_shot_classification, inputs=[zs_in, zs_labels], outputs=zs_out
            )

        # 3. Text Generation
        with gr.Tab("3. Text Generation"):
            tg_in = gr.Textbox(
                label="Incomplete Prompt",
                value="If I continue to complete all in-class exercises in this modern software practices course,",
            )
            tg_max = gr.Slider(10, 60, value=35, step=1, label="Max New Tokens")
            tg_num = gr.Slider(1, 4, value=2, step=1, label="Number of Alternatives")
            tg_out = gr.Textbox(label="Generated Continuations", lines=6)
            gr.Button("Generate Text", variant="primary").click(
                tasks.text_generation, inputs=[tg_in, tg_max, tg_num], outputs=tg_out
            )

        # 4. Mask Filling
        with gr.Tab("4. Mask Filling"):
            fm_in = gr.Textbox(
                label="Masked Sentence (use <mask> or [MASK])",
                value="To understand generative AI, one must study <mask> well.",
            )
            fm_k = gr.Slider(1, 5, value=3, step=1, label="Top K")
            fm_out = gr.Label(label="Predicted Fillings")
            gr.Button("Fill Mask", variant="primary").click(
                tasks.mask_filling, inputs=[fm_in, fm_k], outputs=fm_out
            )

        # 5. Named Entity Recognition
        with gr.Tab("5. NER"):
            ner_in = gr.Textbox(
                label="Input Sentence",
                lines=3,
                value="I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye.",
            )
            ner_out = gr.Textbox(label="Extracted Entities", lines=6)
            gr.Button("Extract Entities", variant="primary").click(
                tasks.named_entity_recognition, inputs=ner_in, outputs=ner_out
            )

        # 6. Question Answering
        with gr.Tab("6. Question Answering"):
            qa_ctx = gr.Textbox(
                label="Context",
                lines=3,
                value="I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye.",
            )
            qa_q = gr.Textbox(
                label="Question",
                value="Which company or institution do they work for?",
            )
            qa_out = gr.Textbox(label="Answer", lines=3)
            gr.Button("Ask Question", variant="primary").click(
                tasks.question_answering, inputs=[qa_ctx, qa_q], outputs=qa_out
            )

        # 7. Text Summarization
        with gr.Tab("7. Summarization"):
            sum_in = gr.Textbox(
                label="Article / Paragraph",
                lines=6,
                value=(
                    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, "
                    "often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States "
                    "housing bubble, its effects rippled across the globe, leading to the collapse of major financial "
                    "institutions and a deep international recession. The crisis began with the subprime mortgage market. "
                    "In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk "
                    "loans to borrowers with poor credit."
                ),
            )
            sum_out = gr.Textbox(label="Summary", lines=4)
            gr.Button("Summarize", variant="primary").click(
                tasks.text_summarization, inputs=sum_in, outputs=sum_out
            )

        # 8. Text Translation
        with gr.Tab("8. Translation (FR -> EN)"):
            tr_in = gr.Textbox(
                label="French Input",
                value="Ce cours est produit par Hugging Face.",
            )
            tr_out = gr.Textbox(label="English Translation")
            gr.Button("Translate", variant="primary").click(
                tasks.text_translation, inputs=tr_in, outputs=tr_out
            )

        # 9. Image Classification
        with gr.Tab("9. Image Classification (ViT)"):
            img_in = gr.Image(type="pil", label="Upload Image")
            img_out = gr.Label(label="Top 5 Predictions (google/vit-base-patch16-224)")
            gr.Button("Classify Image", variant="primary").click(
                tasks.image_classification, inputs=img_in, outputs=img_out
            )

        # 10. Automatic Speech Recognition
        with gr.Tab("10. Speech Recognition (Whisper)"):
            asr_in = gr.Audio(type="filepath", label="Upload or Record Audio")
            asr_out = gr.Textbox(label="Transcription (openai/whisper-large-v3)", lines=4)
            gr.Button("Transcribe Audio", variant="primary").click(
                tasks.automatic_speech_recognition, inputs=asr_in, outputs=asr_out
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
