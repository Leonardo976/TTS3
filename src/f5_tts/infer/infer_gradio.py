import re
import tempfile
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

# Cargar modelos necesarios
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

# Función para traducir números a texto
def traducir_numero_a_texto(texto):
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang="es")

    texto_traducido = re.sub(r"\b\d+\b", reemplazar_numero, texto)
    return texto_traducido

# Función para dividir audio largo en partes
def dividir_audio(audio_path, max_duration=30):
    audio, sr = torchaudio.load(audio_path)
    audio_length = audio.shape[1] / sr
    if audio_length <= max_duration:
        return [audio]
    
    split_audio = []
    samples_per_part = int(sr * max_duration)
    for i in range(0, audio.shape[1], samples_per_part):
        split_audio.append(audio[:, i : i + samples_per_part])
    return split_audio

# Transcribir audio con Whisper
def transcribir_audio(audio_path):
    partes_audio = dividir_audio(audio_path)
    transcripcion = []
    for parte in partes_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            torchaudio.save(temp_audio_file.name, parte, 16000)
            result = whisper_pipeline(temp_audio_file.name)
            transcripcion.append(result["text"])
    return " ".join(transcripcion)

# Función para manejar emociones y generación de texto
def parse_emotions_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_emotion = "Regular"
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            text = token.strip()
            if text:
                segments.append({"emotion": current_emotion, "text": text})
        else:
            current_emotion = token.strip()
    return segments

# Interfaz de Gradio para multi-habla
with gr.Blocks() as app_multihabla:
    gr.Markdown("# Generación de Multi-Habla")

    with gr.Row():
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia Regular", lines=2)

    emotions = gr.State(value={})
    emotion_name = gr.Textbox(label="Nombre de la Emoción")
    emotion_audio = gr.Audio(label="Audio de Referencia para la Emoción", type="filepath")
    add_emotion_btn = gr.Button("Agregar Emoción")

    def add_emotion(emotions, emotion_name, emotion_audio):
        if not emotion_name or not emotion_audio:
            return emotions
        ref_text = transcribir_audio(emotion_audio) if emotion_audio else None
        emotions[emotion_name] = {"audio": emotion_audio, "ref_text": ref_text}
        return emotions

    add_emotion_btn.click(
        add_emotion,
        inputs=[emotions, emotion_name, emotion_audio],
        outputs=[emotions],
    )

    gen_text_input = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Ejemplo: {Alegre} Hola, ¿cómo estás? {Triste} Estoy un poco cansado.",
    )
    generate_btn = gr.Button("Generar Multi-Habla")
    audio_output = gr.Audio(label="Audio Generado")

    def generate_audio(regular_audio, regular_ref_text, gen_text, emotions):
        if not regular_audio or not gen_text:
            return None
        if not regular_ref_text:
            regular_ref_text = transcribir_audio(regular_audio)
        emotions_audio_map = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        emotions_audio_map.update(emotions)
        segments = parse_emotions_text(gen_text)
        generated_audio = []
        for segment in segments:
            emotion = segment["emotion"]
            text = segment["text"]
            if emotion in emotions_audio_map:
                ref_audio = emotions_audio_map[emotion]["audio"]
                ref_text = emotions_audio_map[emotion]["ref_text"]
            else:
                ref_audio = emotions_audio_map["Regular"]["audio"]
                ref_text = emotions_audio_map["Regular"]["ref_text"]
            audio, _ = infer_process(ref_audio, ref_text, text, F5TTS_ema_model, vocoder)
            generated_audio.append(audio[1])
        if generated_audio:
            return 16000, np.concatenate(generated_audio)
        return None

    generate_btn.click(
        generate_audio,
        inputs=[regular_audio, regular_ref_text, gen_text_input, emotions],
        outputs=[audio_output],
    )

# Lanzar aplicación
@click.command()
@click.option("--port", default=7860, type=int)
def main(port):
    app_multihabla.queue().launch(share=True, server_port=port)

if __name__ == "__main__":
    main()
