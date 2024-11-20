# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re
import tempfile

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
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


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

vocoder = load_vocoder()

# Load models
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
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    ema_model = F5TTS_ema_model

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    # Multi-style generation UI
    gr.Markdown(
        """
    # Generación de Múltiples Tipos de Habla

    Esta sección te permite generar múltiples tipos de habla o las voces de múltiples personas. Ingresa tu texto en el formato mostrado a continuación, y el sistema generará el habla utilizando el tipo apropiado. Si no se especifica, el modelo utilizará el tipo de habla regular. El tipo de habla actual se usará hasta que se especifique el siguiente tipo de habla.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Entrada de Ejemplo:**                                                                      
            {Regular} Hola, me gustaría pedir un sándwich, por favor.                                                          \
            {Sorprendido} ¿Qué quieres decir con que no tienen pan?                                                                      
            {Triste} Realmente quería un sándwich...                                                              
            {Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!                                                                       
            {Susurro} Solo volveré a casa y lloraré ahora.                                                                           
            {Gritando} ¿Por qué yo?!                                                                         

            """
        )

    gr.Markdown("Sube diferentes clips de audio para cada tipo de habla. El primer tipo de habla es obligatorio.")

    # Regular speech type (mandatory)
    with gr.Row():
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    # Additional speech types
    max_speech_types = 10
    speech_type_rows = []
    speech_type_names = [regular_name]
    speech_type_audios = []
    speech_type_ref_texts = []

    for _ in range(max_speech_types):
        with gr.Row(visible=False) as row:
            name_input = gr.Textbox(label="Nombre del Tipo de Habla")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)

    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")

    # Button for generating speech
    gen_text_input_multistyle = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="{Regular} Hola, {Triste} estoy cansado.",
    )
    generate_multistyle_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")
    audio_output_multistyle = gr.Audio(label="Audio Sintetizado")

    @gpu_decorator
    def generate_multistyle_speech(regular_audio, regular_ref_text, gen_text, *args):
        # Aquí se implementa la generación de habla multi-estilo con la lógica existente.
        pass

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[regular_audio, regular_ref_text, gen_text_input_multistyle]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts,
        outputs=[audio_output_multistyle],
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
    print("Iniciando la aplicación...")
    app_multistyle.queue(api_open=api).launch(server_name=host, server_port=port, share=True, inbrowser=True, show_api=api)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app_multistyle.queue().launch(share=True, inbrowser=True)
