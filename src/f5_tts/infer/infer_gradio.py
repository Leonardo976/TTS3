import re
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

# Cargar vocoder y modelo
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

def parse_speechtypes_text(gen_text):
    """Divide el texto en segmentos con estilos y velocidades."""
    pattern = r"\{([^:}]+):?([^}]*)\}"  # Captura estilo/velocidad
    tokens = re.split(pattern, gen_text)

    segments = []
    current_style = "Regular"
    current_speed = 1.0

    for i in range(len(tokens)):
        if i % 3 == 0:  # Texto
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "speed": current_speed, "text": text})
        elif i % 3 == 1:  # Estilo o comando
            if tokens[i].lower() == "silencio":
                segments.append({"style": "Silencio", "speed": float(tokens[i + 1] or 1.0), "text": ""})
            else:
                current_style = tokens[i].strip()
        elif i % 3 == 2:  # Velocidad
            current_speed = float(tokens[i] or 1.0)

    return segments

def generate_multistyle_speech(
    regular_audio,
    regular_ref_text,
    gen_text,
    speech_types,
    remove_silence=False,
):
    """Genera audio multihabla basado en los segmentos y configuraciones."""
    segments = parse_speechtypes_text(gen_text)
    generated_audio_segments = []
    current_style = "Regular"

    for segment in segments:
        style = segment["style"]
        speed = segment["speed"]
        text = segment["text"]

        if style == "Silencio":
            silence_duration = int(speed * 24000)  # Convertir segundos a muestras
            generated_audio_segments.append(np.zeros(silence_duration))
            continue

        if style in speech_types:
            current_style = style
        else:
            current_style = "Regular"

        ref_audio = speech_types[current_style]["audio"]
        ref_text = speech_types[current_style].get("ref_text", "")

        audio, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=0.15,
            speed=speed,
            show_info=print,
            progress=gr.Progress(),
        )
        sr, audio_data = audio

        generated_audio_segments.append(audio_data)

    # Concatenar segmentos generados
    if generated_audio_segments:
        final_audio_data = np.concatenate(generated_audio_segments)
        return (sr, final_audio_data)
    else:
        gr.Warning("No se generó ningún audio.")
        return None

with gr.Blocks() as app_multistyle:
    gr.Markdown(
        """
        # Generación de Múltiples Tipos de Habla

        Ingresa el texto en el formato `{Estilo:Velocidad} Texto`. 
        Usa `{silencio:duración_en_segundos}` para agregar pausas.

        **Ejemplo:**
        ```
        {Regular:1.0} Hola, ¿cómo estás? {silencio:0.5}
        {Enojado:1.2} ¡Esto es inaceptable! {silencio:0.2}
        {Susurro:0.8} Por favor, no se lo digas a nadie.
        ```
        """
    )

    regular_audio = gr.Audio(label="Audio de Referencia (Regular)", type="filepath")
    regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    speech_type_inputs = []
    for i in range(4):  # Limitar a 4 estilos adicionales
        with gr.Row(visible=i == 0):
            style_name = gr.Textbox(label=f"Estilo #{i + 1}")
            style_audio = gr.Audio(label=f"Audio de Referencia para Estilo #{i + 1}", type="filepath")
            style_ref_text = gr.Textbox(label=f"Texto de Referencia para Estilo #{i + 1}", lines=2)
            speech_type_inputs.append((style_name, style_audio, style_ref_text))

    gen_text_input = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Escribe el texto con los estilos y velocidades especificados...",
    )
    remove_silence_checkbox = gr.Checkbox(label="Eliminar Silencios", value=False)
    generate_btn = gr.Button("Generar")

    audio_output = gr.Audio(label="Audio Generado")

    def prepare_speech_types(regular_audio, regular_ref_text, *args):
        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        for style_name, style_audio, style_ref_text in zip(*[iter(args)] * 3):
            if style_name and style_audio:
                speech_types[style_name] = {"audio": style_audio, "ref_text": style_ref_text}
        return speech_types

    generate_btn.click(
        lambda regular_audio, regular_ref_text, gen_text, remove_silence, *args: generate_multistyle_speech(
            regular_audio,
            regular_ref_text,
            gen_text,
            prepare_speech_types(regular_audio, regular_ref_text, *args),
            remove_silence,
        ),
        inputs=[regular_audio, regular_ref_text, gen_text_input, remove_silence_checkbox]
        + [item for sublist in speech_type_inputs for item in sublist],
        outputs=audio_output,
    )

if __name__ == "__main__":
    app_multistyle.launch()
