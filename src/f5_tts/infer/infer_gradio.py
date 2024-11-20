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

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave)

def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            style = tokens[i].strip()
            current_style = style

    return segments

with gr.Blocks() as app:
    gr.Markdown(
        """
# Spanish-F5 Multi-Habla

Esta interfaz permite generar múltiples tipos de habla o las voces de múltiples personas. Ingresa tu texto en el formato mostrado a continuación, y el sistema generará el habla utilizando el tipo apropiado. Si no se especifica, el modelo utilizará el tipo de habla regular. El tipo de habla actual se usará hasta que se especifique el siguiente tipo de habla.
"""
    )

    with gr.Row():
        gr.Markdown(
            """
            **Entrada de Ejemplo:**                                                                      
            {Regular} Hola, me gustaría pedir un sándwich, por favor.                                                         
            {Sorprendido} ¿Qué quieres decir con que no tienen pan?                                                                      
            {Triste} Realmente quería un sándwich...                                                              
            {Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!                                                                       
            {Susurro} Solo volveré a casa y lloraré ahora.                                                                           
            {Gritando} ¿Por qué yo?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Entrada de Ejemplo 2:**                                                                                
            {Speaker1_Feliz} Hola, me gustaría pedir un sándwich, por favor.                                                            
            {Speaker2_Regular} Lo siento, nos hemos quedado sin pan.                                                                                
            {Speaker1_Triste} Realmente quería un sándwich...                                                                             
            {Speaker2_Susurro} Te daré el último que estaba escondiendo.                                                                     
            """
        )

    gr.Markdown(
        "Sube diferentes clips de audio para cada tipo de habla. El primer tipo de habla es obligatorio. Puedes agregar tipos de habla adicionales haciendo clic en el botón 'Agregar Tipo de Habla'."
    )

    # Regular speech type (mandatory)
    with gr.Row():
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
            regular_insert = gr.Button("Insertar", variant="secondary")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    # Additional speech types (up to 99 more)
    max_speech_types = 100
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

    # Button to add speech type
    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")

    # Keep track of current number of speech types
    speech_type_count = gr.State(value=0)

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Ingresa el guion con los nombres de los hablantes (o tipos de emociones) al inicio de cada bloque..."
    )

    # Model choice
    model_choice_multistyle = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS")

    with gr.Accordion("Configuraciones Avanzadas", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Eliminar Silencios",
            value=False,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Audio Sintetizado")

    # Function implementations
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

    def make_delete_speech_type_fn(index):
        def delete_speech_type_fn(speech_type_count):
            row_updates = []
            for i in range(max_speech_types - 1):
                if i == index:
                    row_updates.append(gr.update(visible=False))
                else:
                    row_updates.append(gr.update())
            speech_type_count = max(0, speech_type_count - 1)
            return [speech_type_count] + row_updates
        return delete_speech_type_fn

    for i, delete_btn in enumerate(speech_type_delete_btns):
        delete_fn = make_delete_speech_type_fn(i)
        delete_btn.click(delete_fn, inputs=speech_type_count, outputs=[speech_type_count] + speech_type_rows)

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "Ninguno"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return gr.update(value=updated_text)
        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    @gpu_decorator
    def generate_multistyle_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        *args,
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]
        speech_type_audios_list = args[num_additional_speech_types : 2 * num_additional_speech_types]
        speech_type_ref_texts_list = args[2 * num_additional_speech_types : 3 * num_additional_speech_types]
        model_choice = args[3 * num_additional_speech_types]
        remove_silence = args[3 * num_additional_speech_types + 1]

        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}

        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}

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

            ref_audio = speech_types[current_style]["audio"]
            ref_text = speech_types[current_style].get("ref_text", "")

            audio = infer(
                ref_audio, ref_text, text, model_choice, remove_silence, 0, show_info=print
            )
            sr, audio_data = audio
            generated_audio_segments.append(audio_data)

        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("No se generó ningún audio.")
            return None

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            model_choice_multistyle,
            remove_silence_multistyle,
        ],
        outputs=audio_output_multistyle,
    )

    def validate_speech_types(gen_text, regular_name, *args):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]

        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            return gr.update(interactive=False)
        else:
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )

def main():
    vocoder = load_vocoder()

    # load models
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    F5TTS_ema_model = load_model(
        DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
    )

    # Rest of your code remains the same until the app launch...
    
    if __name__ == "__main__":
        if not USING_SPACES:
            app.queue(api_open=True).launch(server_port=None, share=True, show_api=True)
        else:
            app.queue().launch()

# Add this at the very end of the file
if __name__ == "__main__":
    main()