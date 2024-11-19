import re
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from vocos import Vocos
from f5_tts.model import CFM
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    save_spectrogram,
)

# Configuración del vocoder y modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder()

# Configuración del modelo
model_ckpt_path = str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
vocab_file_path = str(cached_path("hf://jpgallegoar/F5-Spanish/vocab.txt"))

F5TTS_model = load_model(
    model_ckpt_path,
    vocab_file_path,
    mel_spec_type="vocos",
    ode_method="euler",
    use_ema=True,
    device=device,
)

# Conversión de números a texto
def traducir_numero_a_texto(texto):
    from num2words import num2words

    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

    return texto_traducido

# Inferencia en Gradio
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence,
    cross_fade_duration=0.15,
    speed=1.0,
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = traducir_numero_a_texto(gen_text)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
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
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Guardar espectrograma
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


# Interfaz de Gradio
with gr.Blocks() as app:
    gr.Markdown("# F5-TTS Multi-Habla en Español")
    with gr.Row():
        ref_audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
        gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    with gr.Accordion("Configuraciones Avanzadas", open=False):
        ref_text_input = gr.Textbox(
            label="Texto de Referencia",
            info="Opcional: Transcribe automáticamente si está vacío.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
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
        )
        cross_fade_duration_slider = gr.Slider(
            label="Duración del Cross-Fade (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
        )
    generate_btn = gr.Button("Generar", variant="primary")
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

# Iniciar Gradio
if __name__ == "__main__":
    app.launch()
