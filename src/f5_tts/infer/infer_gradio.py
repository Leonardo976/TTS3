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
)

# Load vocoder and model
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)


def traducir_numero_a_texto(texto):
    """Convierte números en palabras dentro del texto."""
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')
    return re.sub(r'\b\d+\b', reemplazar_numero, texto)


@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    """Realiza la inferencia de texto a audio."""
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    ema_model = F5TTS_ema_model

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "
    gen_text = traducir_numero_a_texto(gen_text.lower())

    final_wave, final_sample_rate, _ = infer_process(
        ref_audio, ref_text, gen_text, ema_model, vocoder, cross_fade_duration=cross_fade_duration, speed=speed
    )

    # Remove silence if required
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(tmpfile.name)
            final_wave, _ = torchaudio.load(tmpfile.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return final_sample_rate, final_wave


def parse_speechtypes_text(gen_text):
    """Analiza el texto y separa los estilos de habla."""
    tokens = re.split(r"\{(.*?)\}", gen_text)
    segments = []
    current_style = "Regular"
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            if token.strip():
                segments.append({"style": current_style, "text": token.strip()})
        else:
            current_style = token.strip()
    return segments


with gr.Blocks() as app:
    gr.Markdown("# Generación de Múltiples Tipos de Habla")

    with gr.Row():
        regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath", live=True)
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    max_speech_types = 10
    speech_type_count = gr.State(value=0)
    speech_type_rows = []
    speech_type_names = []
    speech_type_audios = []
    speech_type_ref_texts = []

    for _ in range(max_speech_types):
        row = gr.Row(visible=False)
        with row:
            name_input = gr.Textbox(label="Nombre del Tipo de Habla")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath", live=True)
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)

    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    generate_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")
    audio_output = gr.Audio(label="Audio Sintetizado", live=True)

    def toggle_speech_type_row(count):
        """Muestra u oculta filas adicionales de tipos de habla."""
        if count < max_speech_types:
            updates = [gr.update(visible=True) if i == count else gr.update() for i in range(max_speech_types)]
            return count + 1, *updates
        return count, *[gr.update() for _ in range(max_speech_types)]

    add_speech_type_btn.click(toggle_speech_type_row, inputs=[speech_type_count], outputs=[speech_type_count, *speech_type_rows])

    @gpu_decorator
    def generate_multistyle_audio(
        regular_audio, regular_ref_text, gen_text, *args
    ):
        """Genera el audio concatenando segmentos de diferentes estilos."""
        speech_type_names = args[:max_speech_types]
        speech_type_audios = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts = args[2 * max_speech_types :]

        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        for name, audio, ref_text in zip(speech_type_names, speech_type_audios, speech_type_ref_texts):
            if name and audio:
                speech_types[name] = {"audio": audio, "ref_text": ref_text}

        segments = parse_speechtypes_text(gen_text)
        audio_segments = []

        for segment in segments:
            style = segment["style"]
            text = segment["text"]
            if style in speech_types:
                sr, audio = infer(
                    speech_types[style]["audio"], speech_types[style]["ref_text"], text, F5TTS_ema_model, False
                )
                audio_segments.append(audio)

        if audio_segments:
            return sr, np.concatenate(audio_segments)
        else:
            return None

    generate_btn.click(
        generate_multistyle_audio,
        inputs=[regular_audio, regular_ref_text, gen_text_input]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts,
        outputs=audio_output,
    )


@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--share", "-s", default=False, is_flag=True, help="Compartir la aplicación públicamente")
def main(port, share):
    """Lanza la aplicación Gradio."""
    app.queue(api_open=True).launch(server_port=port, share=share, inbrowser=True)


if __name__ == "__main__":
    main()
