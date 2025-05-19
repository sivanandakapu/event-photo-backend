# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os
import uuid
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

origins = [
    "*",  # or specify your domain(s) like "http://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow all origins or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

PHOTO_DB = "photo_db"
TEMP_DIR = "temp_uploads"

os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/find_faces/")
async def find_faces(file: UploadFile = File(...)):
    # Save uploaded selfie temporarily
    selfie_filename = f"{uuid.uuid4()}.jpg"
    selfie_path = os.path.join(TEMP_DIR, selfie_filename)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Search the face in the photo_db directory
        result_df = DeepFace.find(
            img_path=selfie_path,
            db_path=PHOTO_DB,
            enforce_detection=False
        )

        if isinstance(result_df, list):
            result_df = result_df[0]  # DeepFace returns list of DataFrames

        matched_files = result_df['identity'].tolist() if not result_df.empty else []

        return JSONResponse(content={"matches": matched_files})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        os.remove(selfie_path)
