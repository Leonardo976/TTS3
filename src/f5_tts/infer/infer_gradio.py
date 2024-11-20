import re
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    save_spectrogram,
    remove_silence_for_generated_wav,
)

# Configuración inicial
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder()

F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_model = load_model(
    model_cls=torch.nn.Module,  # Ajustar al modelo correcto si aplica
    model_cfg=F5TTS_model_cfg,
    ckpt_path="/path/to/model_1200000.safetensors",
    device=device,
)

# Función para traducir números a texto
def traducir_numero_a_texto(texto):
    from num2words import num2words

    texto_separado = re.sub(r"([A-Za-z])(\d)", r"\1 \2", texto)
    texto_separado = re.sub(r"(\d)([A-Za-z])", r"\1 \2", texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang="es")

    return re.sub(r"\b\d+\b", reemplazar_numero, texto_separado)

# Generación de múltiples tipos de habla
def generate_multistyle_speech(
    regular_audio,
    regular_ref_text,
    gen_text,
    *args,
):
    speech_type_names = args[:99]  # Máximo 99 tipos adicionales
    speech_type_audios = args[99:198]
    speech_type_ref_texts = args[198:297]
    remove_silence = args[297]

    # Preparar los tipos de habla
    speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
    for name, audio, ref_text in zip(speech_type_names, speech_type_audios, speech_type_ref_texts):
        if name and audio:
            speech_types[name] = {"audio": audio, "ref_text": ref_text}

    # Parsear el texto generado
    segments = parse_speechtypes_text(gen_text)

    # Generar audio para cada segmento
    generated_segments = []
    current_style = "Regular"
    for segment in segments:
        style = segment["style"]
        text = segment["text"]

        if style in speech_types:
            current_style = style
        ref_audio = speech_types[current_style]["audio"]
        ref_text = speech_types[current_style].get("ref_text", "")

        audio, _ = infer(
            ref_audio,
            ref_text,
            text,
            F5TTS_model,
            remove_silence,
            cross_fade_duration=0.15,
            speed=1.0,
        )
        generated_segments.append(audio[1])

    # Concatenar los segmentos generados
    final_audio = np.concatenate(generated_segments) if generated_segments else None
    return (24000, final_audio)

# Función para parsear texto de tipos de habla
def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"

    for i, token in enumerate(tokens):
        if i % 2 == 0:  # Texto
            if token.strip():
                segments.append({"style": current_style, "text": token.strip()})
        else:  # Tipo de habla
            current_style = token.strip()

    return segments

# Interfaz de Gradio
with gr.Blocks() as app_multistyle:
    gr.Markdown("# Generación Multi-Habla")
    regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
    regular_ref_text = gr.Textbox(label="Texto de Referencia Regular")
    gen_text_input = gr.Textbox(label="Texto para Generar")
    remove_silence = gr.Checkbox(label="Eliminar Silencios", value=False)

    # Configurar tipos adicionales de habla
    speech_type_names = [gr.Textbox(label=f"Nombre del Tipo {i+1}") for i in range(99)]
    speech_type_audios = [gr.Audio(label=f"Audio del Tipo {i+1}", type="filepath") for i in range(99)]
    speech_type_ref_texts = [gr.Textbox(label=f"Texto del Tipo {i+1}") for i in range(99)]

    # Botón de generación
    generate_btn = gr.Button("Generar")
    audio_output = gr.Audio(label="Audio Generado")

    # Configuración del botón
    generate_btn.click(
        generate_multistyle_speech,
        inputs=[regular_audio, regular_ref_text, gen_text_input]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [remove_silence],
        outputs=[audio_output],
    )

# Lanzar la app
if __name__ == "__main__":
    app_multistyle.launch()
