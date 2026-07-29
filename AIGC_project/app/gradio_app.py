from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_inference_model, predict_image
from src.visualize_fft import fft_spectrum_image, make_side_by_side


DEFAULT_CONFIG = "configs/resnet50_cifake_quick_gpu.yaml"
DEFAULT_CKPT = "results/checkpoints/resnet50_cifake_quick_gpu/best_f1.pth"


def build_app(config_path: str, ckpt_path: str, device: str):
    import gradio as gr

    model, transform, torch_device, _, checkpoint = load_inference_model(config_path, ckpt_path, device)

    def infer(image):
        if image is None:
            return "No image", 0.0, None, "Please upload an image."
        pil_image = Image.fromarray(image).convert("RGB")
        result = predict_image(model, transform, torch_device, pil_image)
        spectrum = fft_spectrum_image(pil_image)
        preview = make_side_by_side(pil_image, spectrum, 256)
        text = (
            f"Prediction: {result['label']}\n"
            f"AI-generated probability: {result['fake_probability']:.4f}\n"
            f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}"
        )
        return result["label"], result["fake_probability"], preview, text

    with gr.Blocks(title="AIGC-Guard") as demo:
        gr.Markdown("# AIGC-Guard")
        with gr.Row():
            image = gr.Image(type="numpy", label="Upload image")
            with gr.Column():
                label = gr.Label(label="Prediction")
                prob = gr.Number(label="AI-generated probability", precision=4)
                preview = gr.Image(type="pil", label="Original + FFT spectrum")
                detail = gr.Textbox(label="Details", lines=4)
        image.change(infer, inputs=image, outputs=[label, prob, preview, detail])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}. Pass --ckpt with a trained checkpoint path."
        )

    app = build_app(args.config, args.ckpt, args.device)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        quiet=False,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
