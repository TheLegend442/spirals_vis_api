# server.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import os
import torch
import tempfile
import uuid
import numpy as np
from pathlib import Path

from chronos_emb import ChronosEmbedder
from transformer import SpiralTransformer, SpiralDataset
from inference import run_inference_and_plot
from helper_functions import preprocess_save_spiral

def load_models():
    
    # Chronos embedder from chronos_emb.py
    chronos = ChronosEmbedder()
    
    # My latest transformer model
    MODEL_PATH = "./models/sp_trans_20260123_014021.pt"
    device = "cpu"
    model = SpiralTransformer(
        embed_dim=768,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.1,
        max_seq_len=1024,
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    return chronos, model

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    chronos, model = load_models()
    app.state.chronos = chronos
    app.state.model = model

    yield
    
app = FastAPI(lifespan=lifespan)

def run_inference_from_csv(csv_path: str, output_path: str, td: str) -> None:

    chronos: ChronosEmbedder = app.state.chronos
    model: SpiralTransformer = app.state.model

    temp_npz_path = os.path.join(td, f"inference_{uuid.uuid4().hex}.npz")
    preprocess_save_spiral(
        input_csv_path=csv_path,
        output_npz_path=temp_npz_path,
        pts_per_rotation=100,
        smoothing=False,
        chronos_pipeline=chronos,
        mirror_over_x=True,
    )

    dataset = SpiralDataset([[Path(temp_npz_path)]])

    with torch.inference_mode():
        run_inference_and_plot(
            model=model,
            dataset=dataset,
            device="cpu",
            thr_tight=2.2,
            plot_title="Spiral Tightening Prediction",
            save_path=output_path,
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thelegend442.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    
    print("Received file:", file.filename, "size:", len(data))

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "input.csv")
        out_path = os.path.join(td, "output.png")

        with open(csv_path, "wb") as f:
            f.write(data)

        run_inference_from_csv(csv_path, out_path, td)

        if not os.path.exists(out_path):
            raise HTTPException(500, "No output produced")

        with open(out_path, "rb") as f:
            png = f.read()

    return Response(content=png, media_type="image/png")