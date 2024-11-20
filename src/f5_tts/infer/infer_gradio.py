import torch
from importlib.resources import files
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
from safetensors.torch import load_file

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuración general
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
ode_method = "euler"


def load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    """
    Carga un punto de control en el modelo.
    """
    if dtype is None:
        dtype = (
            torch.float16 if device == "cuda" and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    """
    Carga el modelo F5-TTS con la configuración adecuada.
    """
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("tokenizer : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    # Obtener el mapeo de vocabulario
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)

    try:
        model = CFM(
            transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
            mel_spec_kwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
            odeint_kwargs=dict(method=ode_method),
            vocab_char_map=vocab_char_map,
        ).to(device)
    except TypeError as e:
        raise TypeError(
            f"Error al inicializar el modelo con los argumentos proporcionados: {e}. "
            "Asegúrate de que el modelo y la configuración coincidan con la implementación esperada."
        )

    # Cargar el punto de control
    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    """
    Procesa el audio de referencia y el texto para prepararlos para la inferencia.
    """
    show_info("Procesando audio de referencia...")
    from pydub import AudioSegment, silence

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short and len(aseg) > 15000:
            aseg = aseg[:15000]  # Limita a 15 segundos
            show_info("Audio truncado a 15 segundos.")

        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        # Aquí puedes implementar un modelo ASR para transcribir automáticamente
        ref_text = "Texto transcrito automáticamente."

    return ref_audio, ref_text


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    target_rms=0.1,
    cross_fade_duration=0.15,
    device=device,
):
    """
    Realiza la inferencia del modelo para generar audio a partir de texto.
    """
    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    gen_text_batches = [gen_text]  # Puedes dividir en lotes si es necesario
    generated_waves = []

    for gen_text in gen_text_batches:
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=convert_char_to_pinyin([gen_text]),
                duration=audio.shape[-1] // hop_length,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )
            generated = generated.to(torch.float32)
            generated_wave = vocoder.decode(generated.permute(0, 2, 1))
            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_waves.append(generated_wave)

    # Combinar los audios generados
    final_wave = np.concatenate(generated_waves)
    return final_wave, target_sample_rate
