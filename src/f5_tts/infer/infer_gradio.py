import gradio as gr
import torch
import torchaudio
import click
import tempfile
import soundfile as sf
from cached_path import cached_path
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Configuración del dispositivo y vocoder
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder()

# Configuración del modelo F5-TTS
model_ckpt_path = str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_model = load_model(
    DiT,
    F5TTS_model_cfg,
    model_ckpt_path,
    device=device,
)

# Función para inferencia
def infer(
    ref_audio_path,
    ref_text,
    gen_text,
    remove_silence,
    cross_fade_duration=0.15,
    speed=1.0,
):
    """
    Realiza la síntesis de voz a partir del audio y texto de referencia.
    """
    # Preprocesar audio y texto de referencia
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

    # Proceso de inferencia
    final_wave, final_sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        F5TTS_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        device=device,
    )

    # Eliminar silencios si es necesario
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Guardar espectrograma
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


# Crear interfaz de Gradio
def create_gradio_app():
    """
    Configura la interfaz de usuario para la aplicación de Gradio.
    """
    with gr.Blocks() as app:
        gr.Markdown("# F5-TTS Synthesizer")
        with gr.Row():
            ref_audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            gen_text_input = gr.Textbox(label="Texto para Generar", lines=5)
        with gr.Accordion("Configuraciones Avanzadas", open=False):
            ref_text_input = gr.Textbox(
                label="Texto de Referencia",
                info="Opcional: Si no se proporciona, será transcrito automáticamente.",
                lines=2,
            )
            remove_silence_checkbox = gr.Checkbox(
                label="Eliminar Silencios",
                info="Elimina silencios largos del audio generado.",
                value=False,
            )
            speed_slider = gr.Slider(
                label="Velocidad",
                minimum=0.3,
                maximum=2.0,
                value=1.0,
                step=0.1,
                info="Ajusta la velocidad del audio generado.",
            )
            cross_fade_slider = gr.Slider(
                label="Duración del Cross-Fade (s)",
                minimum=0.0,
                maximum=1.0,
                value=0.15,
                step=0.01,
                info="Duración del cross-fade entre clips de audio generados.",
            )
        generate_button = gr.Button("Generar", variant="primary")
        audio_output = gr.Audio(label="Audio Generado")
        spectrogram_output = gr.Image(label="Espectrograma")

        # Conectar inferencia con los inputs y outputs
        generate_button.click(
            infer,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                remove_silence_checkbox,
                cross_fade_slider,
                speed_slider,
            ],
            outputs=[audio_output, spectrogram_output],
        )
    return app


# Función principal para ejecutar la aplicación
@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-h", default="0.0.0.0", help="Host para ejecutar la aplicación")
@click.option(
    "--share",
    "-s",
    is_flag=True,
    help="Compartir la aplicación a través de un enlace público con Gradio",
)
def main(port, host, share):
    """
    Lanza la aplicación Gradio.
    """
    app = create_gradio_app()
    app.queue().launch(server_name=host, server_port=port, share=share)


# Punto de entrada
if __name__ == "__main__":
    main()
