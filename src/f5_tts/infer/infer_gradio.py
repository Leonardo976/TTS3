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


# Lista global de tipos de habla guardados
saved_speech_types = [{"name": "Regular", "audio": None, "text": None}]  # Regular incluido por defecto


def save_speech_type(name, audio_path, text):
    """Guarda un tipo de habla en la lista global."""
    if name and name not in [s["name"] for s in saved_speech_types]:
        saved_speech_types.append({"name": name, "audio": audio_path, "text": text})
        return f"Guardado: {name}"
    return "El nombre ya existe o no es válido."


def delete_speech_type(name):
    """Elimina un tipo de habla de la lista global."""
    global saved_speech_types
    saved_speech_types = [s for s in saved_speech_types if s["name"] != name]
    return f"Eliminado: {name}"


def update_saved_speech_types():
    """Actualiza la lista de tipos guardados en el menú desplegable."""
    return gr.update(choices=[s["name"] for s in saved_speech_types])


def generate_text_with_type(name):
    """Genera un bloque de texto con un tipo de habla seleccionado."""
    return f"{{{name}}} "


with gr.Blocks() as app:
    gr.Markdown("# Generación de Múltiples Tipos de Habla")

    # Regular speech type (always visible)
    with gr.Row():
        regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla", interactive=False)
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)
        regular_save_btn = gr.Button("Guardar")
        regular_delete_btn = gr.Button("Eliminar", interactive=False)  # Regular no se puede eliminar

    max_speech_types = 10
    speech_type_rows = []
    speech_type_names = []
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_save_btns = []
    speech_type_delete_btns = []

    # Dynamic speech types
    for _ in range(max_speech_types):
        row = gr.Row(visible=False)
        with row:
            name_input = gr.Textbox(label="Nombre del Tipo de Habla")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
            save_btn = gr.Button("Guardar")
            delete_btn = gr.Button("Eliminar")
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_save_btns.append(save_btn)
        speech_type_delete_btns.append(delete_btn)

    saved_speech_types_dropdown = gr.Dropdown(label="Seleccionar Tipo Guardado", choices=[], interactive=True)
    add_text_with_speech_type_btn = gr.Button("Agregar Texto con Tipo de Habla")
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    generate_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")
    audio_output = gr.Audio(label="Audio Sintetizado")
    progress_bar = gr.Textbox(label="Progreso", interactive=False)

    def toggle_speech_type_row(index):
        """Muestra u oculta filas adicionales de tipos de habla."""
        return gr.update(visible=True)

    for i, (save_btn, delete_btn) in enumerate(zip(speech_type_save_btns, speech_type_delete_btns)):
        save_btn.click(
            save_speech_type,
            inputs=[speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
            outputs=progress_bar,
            postprocess=update_saved_speech_types,
        )
        delete_btn.click(
            delete_speech_type,
            inputs=speech_type_names[i],
            outputs=progress_bar,
            postprocess=update_saved_speech_types,
        )

    regular_save_btn.click(
        save_speech_type,
        inputs=[regular_name, regular_audio, regular_ref_text],
        outputs=progress_bar,
        postprocess=update_saved_speech_types,
    )

    add_text_with_speech_type_btn.click(
        generate_text_with_type,
        inputs=saved_speech_types_dropdown,
        outputs=gen_text_input,
    )

    @gpu_decorator
    def generate_multistyle_audio(
        regular_audio, regular_ref_text, gen_text, progress, *args
    ):
        """Genera el audio concatenando segmentos de diferentes estilos."""
        progress.append("Inicio de la generación...")
        speech_type_names = args[:max_speech_types]
        speech_type_audios = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts = args[2 * max_speech_types :]

        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}
        for name, audio, ref_text in zip(speech_type_names, speech_type_audios, speech_type_ref_texts):
            if name and audio:
                speech_types[name] = {"audio": audio, "ref_text": ref_text}

        segments = parse_speechtypes_text(gen_text)
        audio_segments = []

        for i, segment in enumerate(segments):
            style = segment["style"]
            text = segment["text"]
            progress.append(f"Procesando segmento {i + 1}/{len(segments)}: {style}")
            if style in speech_types:
                sr, audio = infer(
                    speech_types[style]["audio"], speech_types[style]["ref_text"], text, F5TTS_ema_model, False
                )
                audio_segments.append(audio)

        progress.append("Finalizando la generación...")
        if audio_segments:
            return sr, np.concatenate(audio_segments)
        else:
            return None

    generate_btn.click(
        generate_multistyle_audio,
        inputs=[regular_audio, regular_ref_text, gen_text_input, progress_bar]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts,
        outputs=[audio_output, progress_bar],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación")
@click.option(
    "--share",
    "-s",
    default=True,
    is_flag=True,
    help="Siempre habilitar el enlace live (Gradio public URL).",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    """
    Ejecuta la aplicación Multi-Habla con las configuraciones de Gradio.
    """
    print("Iniciando la aplicación...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=True,  # Siempre habilitar el live
        show_api=api,
    )


if __name__ == "__main__":
    main()
