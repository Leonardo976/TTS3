import re
import tempfile
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
    save_spectrogram,
)

# Configuración del vocoder y modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder()

# Configuración del modelo
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_model = load_model(
    DiT,
    F5TTS_model_cfg,
    str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors")),
)

# Conversión de números a texto
def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    return re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

# Función para procesar multi-habla
def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"  # Encuentra {tipo}
    tokens = re.split(pattern, gen_text)

    segments = []
    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:  # Texto
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:  # Tipo de habla
            current_style = tokens[i].strip()

    return segments

def generate_multistyle_speech(
    ref_audio,
    ref_text,
    gen_text,
    speech_type_audios,
    speech_type_texts,
    remove_silence,
    speed,
):
    # Crear diccionario de tipos de habla
    speech_types = {"Regular": {"audio": ref_audio, "text": ref_text}}
    for name, audio, text in zip(speech_type_audios.keys(), speech_type_audios.values(), speech_type_texts.values()):
        if audio:
            speech_types[name] = {"audio": audio, "text": text}

    # Procesar segmentos de multi-habla
    segments = parse_speechtypes_text(gen_text)
    generated_audio_segments = []
    current_style = "Regular"

    for segment in segments:
        style = segment["style"]
        text = segment["text"]

        if style in speech_types:
            current_style = style
        else:
            current_style = "Regular"

        audio_data = speech_types[current_style]["audio"]
        text_data = speech_types[current_style].get("text", "")

        audio, _ = infer_process(
            audio_data, text_data, text, F5TTS_model, vocoder, speed=speed, remove_silence=remove_silence, device=device
        )
        sr, audio_wave = audio
        generated_audio_segments.append(audio_wave)

    final_audio_data = np.concatenate(generated_audio_segments)
    return sr, final_audio_data

# Interfaz Gradio
with gr.Blocks() as app_multistyle:
    gr.Markdown("# Generación de Múltiples Tipos de Habla")
    ref_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
    ref_text = gr.Textbox(label="Texto de Referencia Regular", lines=2)
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)

    max_speech_types = 10
    speech_type_names = []
    speech_type_audios = []
    speech_type_texts = []

    for i in range(max_speech_types):
        with gr.Row(visible=False) as row:
            name_input = gr.Textbox(label="Nombre del Tipo de Habla")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_texts.append(text_input)

    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")
    generate_btn = gr.Button("Generar", variant="primary")
    output_audio = gr.Audio(label="Audio Sintetizado")

    # Lógica para generar multi-habla
    generate_btn.click(
        generate_multistyle_speech,
        inputs=[
            ref_audio,
            ref_text,
            gen_text_input,
            {f"speech_type_{i}": audio for i, audio in enumerate(speech_type_audios)},
            {f"text_{i}": text for i, text in enumerate(speech_type_texts)},
        ],
        outputs=output_audio,
    )

# Ejecutar la aplicación
if __name__ == "__main__":
    app_multistyle.launch()
