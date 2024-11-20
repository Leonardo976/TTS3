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

vocoder = load_vocoder()

# load models
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

def parse_text_marks(text):
    segments = []
    pattern = r'\{([^}]+)\}'
    
    # Split text by marks
    parts = re.split(pattern, text)
    
    current_marks = {
        'speed': 1.0,
        'pitch': 1.0,
        'style': 'Regular'
    }
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Text content
            if part.strip():
                segments.append({
                    'text': part.strip(),
                    'marks': current_marks.copy()
                })
        else:  # Mark definition
            marks = part.lower().split()
            for mark in marks:
                if mark == 'fast':
                    current_marks['speed'] = 1.5
                elif mark == 'slow':
                    current_marks['speed'] = 0.7
                elif mark == 'high':
                    current_marks['pitch'] = 1.2
                elif mark == 'low':
                    current_marks['pitch'] = 0.8
                elif mark in ['regular', 'sorprendido', 'triste', 'enojado', 'susurro', 'gritando']:
                    current_marks['style'] = mark.capitalize()
                else:
                    current_marks['style'] = mark
    
    return segments

def infer_with_marks(ref_audio, ref_text, text, model, remove_silence=False, show_info=print):
    segments = parse_text_marks(text)
    audio_segments = []
    
    for segment in segments:
        text_segment = segment['text']
        speed = segment['marks']['speed']
        # Note: pitch modification would need to be implemented in the audio processing pipeline
        
        audio, _ = infer(
            ref_audio,
            ref_text,
            text_segment,
            model,
            remove_silence,
            cross_fade_duration=0.15,
            speed=speed,
            show_info=show_info
        )
        sr, audio_data = audio
        audio_segments.append(audio_data)
    
    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        return (sr, final_audio)
    return None

with gr.Blocks() as app:
    gr.Markdown("""
    # Spanish F5 TTS - Multi-Habla con Marcas de Texto
    
    Esta interfaz permite generar voz con múltiples estilos y marcas de texto.
    
    **Marcas disponibles:**
    - Velocidad: {fast}, {slow}
    - Tono: {high}, {low}
    - Estilos: {Regular}, {Sorprendido}, {Triste}, {Enojado}, {Susurro}, {Gritando}
    
    **Ejemplo:**
    {Regular} Hola, esto es una prueba. {fast} Esta parte es más rápida. {slow} Y esta más lenta.
    {Sorprendido high} ¡Wow, esto es increíble! {Susurro low} Y esto es un secreto...
    """)

    # Regular speech type (mandatory)
    with gr.Row():
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
            regular_insert = gr.Button("Insertar", variant="secondary")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    # Additional speech types
    max_speech_types = 6  # Reduced number of speech types
    speech_type_rows = []
    speech_type_names = [regular_name]
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_delete_btns = []
    speech_type_insert_btns = []
    speech_type_insert_btns.append(regular_insert)

    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Nombre del Tipo de Habla")
                delete_btn = gr.Button("Eliminar", variant="secondary")
                insert_btn = gr.Button("Insertar", variant="secondary")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")
    speech_type_count = gr.State(value=0)

    def add_speech_type_fn(speech_type_count):
        if speech_type_count < max_speech_types - 1:
            speech_type_count += 1
            row_updates = []
            for i in range(max_speech_types - 1):
                if i < speech_type_count:
                    row_updates.append(gr.update(visible=True))
                else:
                    row_updates.append(gr.update())
        else:
            row_updates = [gr.update() for _ in range(max_speech_types - 1)]
        return [speech_type_count] + row_updates

    add_speech_type_btn.click(
        add_speech_type_fn,
        inputs=speech_type_count,
        outputs=[speech_type_count] + speech_type_rows
    )

    gen_text_input = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Ingresa el texto con marcas de estilo y velocidad. Ejemplo:\n{Regular} Hola, esto es {fast} una prueba rápida {slow} y esto va más lento."
    )

    with gr.Accordion("Configuraciones Avanzadas", open=False):
        remove_silence = gr.Checkbox(
            label="Eliminar Silencios",
            value=False,
        )

    generate_btn = gr.Button("Generar Audio", variant="primary")
    audio_output = gr.Audio(label="Audio Generado")

    def generate_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        remove_silence,
        *args
    ):
        if not regular_audio or not gen_text.strip():
            return None

        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        
        # Process additional speech types
        num_additional = max_speech_types - 1
        names = args[:num_additional]
        audios = args[num_additional:2*num_additional]
        ref_texts = args[2*num_additional:3*num_additional]
        
        for name, audio, ref_text in zip(names, audios, ref_texts):
            if name and audio:
                speech_types[name] = {"audio": audio, "ref_text": ref_text}

        segments = parse_text_marks(gen_text)
        audio_segments = []
        
        for segment in segments:
            style = segment['marks']['style']
            if style not in speech_types:
                style = "Regular"
                
            ref_audio = speech_types[style]["audio"]
            ref_text = speech_types[style].get("ref_text", "")
            
            text = segment['text']
            speed = segment['marks']['speed']
            
            audio = infer(
                ref_audio,
                ref_text,
                text,
                "F5-TTS",
                remove_silence,
                cross_fade_duration=0.15,
                speed=speed,
                show_info=print
            )
            sr, audio_data = audio[0]
            audio_segments.append(audio_data)

        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            return (sr, final_audio)
        return None

    generate_btn.click(
        generate_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input,
            remove_silence,
        ] + speech_type_names + speech_type_audios + speech_type_ref_texts,
        outputs=audio_output,
    )

if __name__ == "__main__":
    app.launch()