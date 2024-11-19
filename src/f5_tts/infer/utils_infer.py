import re
import tempfile
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment, silence
from vocos import Vocos
from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parámetros generales
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256

# Función para cargar el vocoder
def load_vocoder():
    print("Cargando vocoder desde HuggingFace...")
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    return vocoder

# Función para cargar el modelo
def load_model(ckpt_path, vocab_file):
    print("Cargando modelo...")
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_num_embeds=vocab_size, mel_dim=n_mel_channels)
    model = CFM(
        transformer=CFM.Transformer(**model_cfg),
        mel_spec_kwargs=dict(n_fft=1024, hop_length=hop_length, n_mel_channels=n_mel_channels, target_sample_rate=24000),
        vocab_char_map=vocab_char_map,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device)

# Procesamiento de texto para estilos de habla
def parse_speechtypes_text(gen_text):
    pattern = r"\{([^:}]+):?([^}]*)\}"  # Captura estilo/velocidad
    tokens = re.split(pattern, gen_text)

    segments = []
    current_style = "Regular"
    current_speed = 1.0

    for i in range(len(tokens)):
        if i % 3 == 0:  # Texto
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "speed": current_speed, "text": text})
        elif i % 3 == 1:  # Estilo
            if tokens[i].lower() == "silencio":
                segments.append({"style": "Silencio", "speed": float(tokens[i + 1] or 1.0), "text": ""})
            else:
                current_style = tokens[i].strip()
        elif i % 3 == 2:  # Velocidad
            current_speed = float(tokens[i] or 1.0)

    return segments

# Proceso de inferencia por lotes
def infer_process(ref_audio, ref_text, gen_text, model, vocoder):
    audio, sr = torchaudio.load(ref_audio)
    audio = audio.to(device)
    segments = parse_speechtypes_text(gen_text)

    generated_waves = []
    for segment in segments:
        if segment["style"] == "Silencio":
            silence_duration = int(segment["speed"] * target_sample_rate)
            generated_waves.append(np.zeros(silence_duration))
            continue

        # Preparación del texto
        text = convert_char_to_pinyin([ref_text + segment["text"]])

        # Inferencia del modelo
        with torch.no_grad():
            mel_spec = model.sample(audio, text, speed=segment["speed"])
            wave = vocoder.decode(mel_spec.permute(0, 2, 1)).squeeze().cpu().numpy()
            generated_waves.append(wave)

    # Concatenación final
    final_wave = np.concatenate(generated_waves)
    return final_wave, target_sample_rate

# Guardar el espectrograma
def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
