# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import os
import re
import tempfile

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from num2words import num2words

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Silenciar logs de TensorFlow para evitar mensajes excesivos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Cargar vocoder
vocoder = load_vocoder()

# Configuración del modelo F5-TTS
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    return texto_traducido

@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, remove_silence, cross_fade_duration=0.15, speed=1.0, show_info=gr.Info
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = traducir_numero_a_texto(gen_text.lower())

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        F5TTS_ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


# Definir la aplicación como una variable global
app = gr.Blocks()

with app:
    gr.Markdown("# F5-TTS en Español")
    ref_audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    generate_btn = gr.Button("Generar", variant="primary")

    with gr.Accordion("Configuraciones Avanzadas", open=False):
        ref_text_input = gr.Textbox(
            label="Texto de Referencia (opcional)",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Eliminar Silencios",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Velocidad",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
        )
        cross_fade_duration_slider = gr.Slider(
            label="Duración de Cross-Fade (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
        )

    audio_output = gr.Audio(label="Audio Generado")
    spectrogram_output = gr.Image(label="Espectrograma")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output],
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Compartir la aplicación a través de un enlace compartido de Gradio",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    global app
    print("Iniciando la aplicación...")
    # Asegúrate de que `share=True`
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=True, show_api=api)


if __name__ == "__main__":
    main()
