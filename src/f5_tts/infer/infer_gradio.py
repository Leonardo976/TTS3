import re
import tempfile
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from num2words import num2words
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

# Cargar el vocoder
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

def infer_audio(ref_audio, ref_text, gen_text, remove_silence):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    gen_text = traducir_numero_a_texto(gen_text)
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        F5TTS_ema_model,
        vocoder,
        remove_silence=remove_silence,
        cross_fade_duration=0.15,
        speed=1.0,
    )
    return final_sample_rate, final_wave

def generate_speech(ref_audio, ref_text, gen_text, remove_silence):
    if not ref_audio or not gen_text.strip():
        return None
    audio = infer_audio(ref_audio, ref_text, gen_text, remove_silence)
    return audio

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# F5-TTS - Synthesizer")

        with gr.Row():
            ref_audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        gen_text_input = gr.Textbox(label="Texto para Generar", lines=5)
        remove_silence = gr.Checkbox(label="Eliminar Silencios", value=False)
        generate_btn = gr.Button("Generar", variant="primary")
        audio_output = gr.Audio(label="Audio Generado")

        generate_btn.click(
            generate_speech,
            inputs=[ref_audio_input, ref_text_input, gen_text_input, remove_silence],
            outputs=audio_output,
        )

    return app

@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-h", default="0.0.0.0", help="Host para ejecutar la aplicación")
@click.option("--share", "-s", is_flag=True, help="Compartir la aplicación públicamente usando Gradio")
def main(port, host, share):
    app = create_app()
    app.queue().launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    main()
