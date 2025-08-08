# backend/main.py
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os

from backend.utils.executive_summary import generate_summary

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(file_path)
    summary_df, stats, warnings = generate_summary(df)

    # Save summary CSV
    summary_csv_path = os.path.join(UPLOAD_DIR, "executive_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    return templates.TemplateResponse(
        "summary.html",
        {
            "request": request,
            "summary": summary_df.to_html(classes="table table-striped", index=False),
            "stats": stats,
            "warnings": warnings,
        },
    )
