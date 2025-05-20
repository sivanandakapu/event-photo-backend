from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

import boto3
import os
import shutil
import uuid
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# === Constants ===
BUCKET_NAME = S3_BUCKET_NAME  # use value from env
PHOTO_DB = "photo_db"
TEMP_DIR = "temp_uploads"

# === Create directories ===
os.makedirs(PHOTO_DB, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# === Initialize FastAPI ===
app = FastAPI()

# === CORS Configuration ===
origins = ["*"]  # For dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === S3 Client ===
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


def sync_s3_photos():
    """
    Download only new images from S3 to the local PHOTO_DB folder.
    """
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    if "Contents" not in response:
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        local_path = os.path.join(PHOTO_DB, os.path.basename(key))

        if not os.path.exists(local_path):
            print(f"Downloading {key}...")
            s3.download_file(BUCKET_NAME, key, local_path)


@app.post("/find_faces/")
async def find_faces(file: UploadFile = File(...)):
    # Step 1: Sync local folder with S3
    try:
        sync_s3_photos()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to sync from S3: {str(e)}"})

    # Step 2: Save uploaded selfie temporarily
    selfie_filename = f"{uuid.uuid4()}.jpg"
    selfie_path = os.path.join(TEMP_DIR, selfie_filename)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 3: Face matching with DeepFace
    try:
        result_df = DeepFace.find(
            img_path=selfie_path,
            db_path=PHOTO_DB,
            enforce_detection=False
        )

        if isinstance(result_df, list):
            result_df = result_df[0]

        matched_files = result_df['identity'].tolist() if not result_df.empty else []

        return JSONResponse(content={"matches": matched_files})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(selfie_path):
            os.remove(selfie_path)
