from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile

app = FastAPI()
model = load_model("cough_cnn_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    audio, sr = librosa.load(path, sr=22050)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    prediction = model.predict(mfcc)
    result = int(np.argmax(prediction))

    return {"prediction": result}