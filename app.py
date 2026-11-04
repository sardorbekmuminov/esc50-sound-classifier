import json
import gradio as gr
import librosa
import numpy as np
import timm
import torch

SR, DURATION = 22050, 5
N_SAMPLES = SR * DURATION
N_FFT, HOP, N_MELS = 1024, 512, 128

CLASSES = json.load(open("classes.json"))
model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASSES), in_chans=1)
model.load_state_dict(torch.load("fine_tuned_resnet18.pt", map_location="cpu"))
model.eval()


def fix_length(sig):
    if len(sig) < N_SAMPLES:
        sig = np.pad(sig, (0, N_SAMPLES - len(sig)))
    return sig[:N_SAMPLES]


def to_mel(sig):
    m = librosa.feature.melspectrogram(y=sig, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    db = librosa.power_to_db(m, ref=np.max)
    return ((db + 40) / 20).astype(np.float32)


@torch.no_grad()
def classify(path):
    if path is None:
        return None
    sig, _ = librosa.load(path, sr=SR, mono=True)
    chunks = [sig[i:i + N_SAMPLES] for i in range(0, max(len(sig), 1), N_SAMPLES)]
    if len(chunks) > 1 and len(chunks[-1]) < SR:   # drop a tiny trailing chunk
        chunks = chunks[:-1]
    x = torch.from_numpy(np.stack([to_mel(fix_length(c)) for c in chunks]))[:, None]
    probs = model(x).softmax(1).mean(0)
    return {c: float(p) for c, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (upload or record)"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 predictions (confidence)"),
    title="Environmental Sound Classifier (ESC-50)",
    description="ResNet18 fine-tuned on log-mel spectrograms. Recognizes 50 everyday sound classes "
                "(dog, rain, siren, keyboard typing, ...). Audio longer than 5 s is split into windows and averaged.",
)

if __name__ == "__main__":
    demo.launch()
