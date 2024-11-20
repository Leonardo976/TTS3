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
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1
):
    """Realiza la inferencia de texto a audio."""
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)
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


# Lista global de tipos de habla guardados
saved_speech_types = [{"name": "Regular", "audio": None, "text": None}]  # Regular incluido por defecto


def save_speech_type(name, audio_path, text):
    """Guarda un tipo de habla en la lista global."""
    if name and name not in [s["name"] for s in saved_speech_types]:
        saved_speech_types.append({"name": name, "audio": audio_path, "text": text})
        return f"Guardado: {name}", update_speech_types()
    return "El nombre ya existe o no es válido.", update_speech_types()


def delete_speech_type(name):
    """Elimina un tipo de habla de la lista global."""
    global saved_speech_types
    if name != "Regular":  # Proteger el tipo "Regular"
        saved_speech_types = [s for s in saved_speech_types if s["name"] != name]
        return f"Eliminado: {name}", update_speech_types()
    return "No se puede eliminar el tipo Regular.", update_speech_types()


def update_speech_types():
    """Devuelve una lista actualizada de los nombres de tipos de habla."""
    return "\n".join(f"- {s['name']}" for s in saved_speech_types)


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

    # Dynamic types of speech display
    speech_types_display = gr.Markdown(value=update_speech_types(), label="Tipos de Habla Guardados")

    remove_silence_checkbox = gr.Checkbox(value=False, label="Eliminar Silencios")
    new_name = gr.Textbox(label="Nombre del Nuevo Tipo de Habla")
    new_audio = gr.Audio(label="Audio del Nuevo Tipo de Habla", type="filepath")
    new_text = gr.Textbox(label="Texto de Referencia del Nuevo Tipo de Habla")
    add_speech_type_btn = gr.Button("Agregar Nuevo Tipo de Habla")
    delete_name = gr.Textbox(label="Nombre del Tipo de Habla a Eliminar")
    delete_btn = gr.Button("Eliminar Tipo de Habla")

    saved_speech_types_dropdown = gr.Dropdown(
        label="Seleccionar Tipo Guardado",
        choices=[s["name"] for s in saved_speech_types],
        interactive=True,
    )
    add_text_with_speech_type_btn = gr.Button("Agregar Texto con Tipo de Habla")
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    generate_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")
    audio_output = gr.Audio(label="Audio Sintetizado")
    progress_bar = gr.Textbox(label="Progreso", interactive=True, lines=5)

    # Save regular type
    regular_save_btn.click(
        save_speech_type,
        inputs=[regular_name, regular_audio, regular_ref_text],
        outputs=[progress_bar, speech_types_display],
    )

    # Add new speech type
    add_speech_type_btn.click(
        save_speech_type,
        inputs=[new_name, new_audio, new_text],
        outputs=[progress_bar, speech_types_display],
    )

    # Delete speech type
    delete_btn.click(
        delete_speech_type,
        inputs=delete_name,
        outputs=[progress_bar, speech_types_display],
    )

    # Add text block with selected speech type
    add_text_with_speech_type_btn.click(
        generate_text_with_type,
        inputs=saved_speech_types_dropdown,
        outputs=gen_text_input,
    )

    @gpu_decorator
    def generate_multistyle_audio(
        regular_audio, regular_ref_text, gen_text, remove_silence, *args
    ):
        """Genera el audio concatenando segmentos de diferentes estilos."""
        speech_types = {s["name"]: {"audio": s["audio"], "ref_text": s["text"]} for s in saved_speech_types}

        segments = re.split(r"\{(.*?)\}", gen_text)
        audio_segments = []

        for i, segment in enumerate(segments):
            style = segment if i % 2 else "Regular"
            if style in speech_types and speech_types[style]["audio"]:
                sr, audio = infer(
                    speech_types[style]["audio"], speech_types[style]["ref_text"], segment, F5TTS_ema_model, remove_silence
                )
                audio_segments.append(audio)

        if audio_segments:
            return sr, np.concatenate(audio_segments)
        return None

    generate_btn.click(
        generate_multistyle_audio,
        inputs=[regular_audio, regular_ref_text, gen_text_input, remove_silence_checkbox],
        outputs=audio_output,
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
