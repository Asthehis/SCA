from fastapi import FastAPI, Request, Form, Depends, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware
import os
import subprocess
import json

from IHM_.IHM.database import get_db, Base, engine
from IHM_.IHM.auth import authenticate_user

app = FastAPI()

current_dir = os.path.dirname(__file__)
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)
app.add_middleware(SessionMiddleware, secret_key='secret_key')
Base.metadata.create_all(bind=engine)

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, username, password)
    if user:
        request.session['user'] = user.nom
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Identifiants incorrects"})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    save_path = f"data/raw/{file.filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f_out:
        f_out.write(await file.read())

    # Écrire le nom du dernier fichier dans un fichier temporaire
    with open("data/last_filename.txt", "w", encoding="utf-8") as f:
        f.write(file.filename)
    print(f"Fichier reçu : {file.filename} Taille : {file.size}")
    return JSONResponse({"status": "Fichier reçu"})

@app.post("/run-step/")
async def run_step(step: str = Query(...)):
    print(f"➡️ Étape demandée : {step}")
    try:
        if step == "transcription":
            result = subprocess.run(["python", "models/transcribe_diarize.py"], capture_output=True, text=True, check=True)
        elif step == "comprehension":
            result = subprocess.run(["python", "models/comprehension.py"], capture_output=True, text=True, check=True)
        elif step == "score":
            result = subprocess.run(["python", "models/score.py"], capture_output=True, text=True, check=True)
        else:
            return JSONResponse({"error": f"Étape inconnue : {step}"})

        return JSONResponse({"output": result.stdout})

    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": e.stderr})

@app.get("/get-score/")
async def get_score():
    try:
        with open("data/score_final.txt", "r", encoding="utf-8") as f:
            score = int(f.read().strip())
        return {"score": score}
    except Exception as e:
        return {"error": str(e)}

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()  # Supprime toutes les données de session
    return RedirectResponse(url="/", status_code=302)
