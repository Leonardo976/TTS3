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

def parse_emotions_text(gen_text):
    pattern = r"\{(.*?)\}"  # Capturar emociones en formato {Emotion}
    tokens = re.split(pattern, gen_text)
    segments = []
    current_emotion = "Regular"  # Emoción predeterminada

    for i, token in enumerate(tokens):
        if i % 2 == 0:  # Texto
            text = token.strip()
            if text:
                segments.append({"emotion": current_emotion, "text": text})
        else:  # Emoción
            current_emotion = token.strip()

    return segments

def infer_multi_emotion(
    ref_audio_orig,
    ref_text,
    gen_text,
    emotions_audio_map,
    remove_silence=False,
    show_info=print,
):
    segments = parse_emotions_text(gen_text)
    generated_audio = []

    for segment in segments:
        emotion = segment["emotion"]
        text = segment["text"]

        if emotion in emotions_audio_map:
            ref_audio = emotions_audio_map[emotion]["audio"]
            ref_text_emotion = emotions_audio_map[emotion].get("ref_text", ref_text or "Texto predeterminado.")
        else:  # Emoción no encontrada, usar "Regular"
            ref_audio = emotions_audio_map["Regular"]["audio"]
            ref_text_emotion = emotions_audio_map["Regular"].get("ref_text", ref_text or "Texto predeterminado.")

        audio, _ = infer_process(
            ref_audio,
            ref_text_emotion,
            text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=0.15,
            speed=1.0,
            show_info=show_info,
        )
        sr, audio_data = audio
        generated_audio.append(audio_data)

    if generated_audio:
        final_audio = np.concatenate(generated_audio)
        return sr, final_audio
    return None

# Construir la interfaz Gradio
with gr.Blocks() as app_tts_multihabla:
    gr.Markdown("# F5-TTS en Español - Multi-Habla")

    # Tipo de habla regular obligatorio
    with gr.Row():
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia Regular", lines=2)

    # Emociones adicionales
    emotions = gr.State(value={})  # Guardar audios y textos asociados a emociones
    emotion_name = gr.Textbox(label="Nombre de la Emoción (ej: Alegre)", placeholder="Escribe el nombre de la emoción")
    emotion_audio = gr.Audio(label="Audio de Referencia para la Emoción", type="filepath")
    emotion_ref_text = gr.Textbox(label="Texto de Referencia para la Emoción", lines=2)
    enable_emotion_add = gr.Checkbox(label="Habilitar Agregar Emoción", value=False)
    add_emotion_btn = gr.Button("Agregar Emoción")

    def add_emotion(emotions, enable, emotion_name, emotion_audio, emotion_ref_text):
        if not enable or not emotion_name or not emotion_audio:
            return gr.update(), gr.update(), emotions  # No hacer nada si no está habilitado o faltan datos
        emotions[emotion_name] = {"audio": emotion_audio, "ref_text": emotion_ref_text}
        return gr.update(value=""), gr.update(value=None), emotions

    add_emotion_btn.click(
        add_emotion,
        inputs=[emotions, enable_emotion_add, emotion_name, emotion_audio, emotion_ref_text],
        outputs=[emotion_name, emotion_audio, emotions],
    )

    # Texto y botón para generar
    gen_text_input = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Ejemplo: {Alegre} Hola, ¿cómo estás? {Triste} Estoy un poco cansado."
    )
    remove_silence_checkbox = gr.Checkbox(label="Eliminar Silencios", value=False)
    generate_btn = gr.Button("Generar Multi-Habla", variant="primary")
    download_btn = gr.Button("Guardar Audio Generado")
    audio_output = gr.Audio(label="Audio Generado")

    def generate_audio(regular_audio, regular_ref_text, gen_text, emotions, remove_silence):
        if not regular_audio or not gen_text:
            return None  # No generar si faltan datos básicos

        emotions_audio_map = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        emotions_audio_map.update(emotions)  # Agregar emociones personalizadas

        result = infer_multi_emotion(
            regular_audio,
            regular_ref_text,
            gen_text,
            emotions_audio_map,
            remove_silence=remove_silence,
        )
        if result:
            sr, audio = result
            output_path = tempfile.mktemp(suffix=".wav")
            sf.write(output_path, audio, sr)
            return sr, audio, output_path
        return None, None, None

    result_audio = gr.State(value=None)

    generate_btn.click(
        generate_audio,
        inputs=[regular_audio, regular_ref_text, gen_text_input, emotions, remove_silence_checkbox],
        outputs=[audio_output, result_audio],
    )

    def save_audio(result_audio):
        if result_audio:
            return gr.File.update(value=result_audio)
        return None

    download_btn.click(
        save_audio,
        inputs=[result_audio],
        outputs=[gr.File(label="Descargar Archivo")],
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
    """
    Ejecuta la aplicación Multi-Habla con las configuraciones de Gradio.
    """
    print("Iniciando la aplicación...")
    app_tts_multihabla.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
    )

if __name__ == "__main__":
    main()
