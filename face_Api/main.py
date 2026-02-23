import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from face_Api.recognizer import load_known_faces, recognize_face_from_image
from face_Api.recognizer import save_known_face, save_faces_from_video
from face_Api import users
from fastapi import HTTPException, Form

app = FastAPI(title="Face Recognition API")

# Permitir acesso do frontendjkdfvbb nbieafcaqb
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ou substitua por ['http://localhost:3000'] se for usar com Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)shdvbcfckweuigfq bfcba\eafbqcileqfb

# Carregar rostos conhecidos
load_known_faces()

@app.post("/reconhecer")
async def reconhecer(arquivo: UploadFile = File(...)):
    image_bytes = await arquivo.read()
    result = recognize_face_from_image(image_bytes)
    return {"resultados": result}


@app.post("/cadastrar")
async def cadastrar(arquivo: UploadFile = File(...), user_id: str | None = Form(None)):
    """Recebe um arquivo de imagem e cadastra como rosto conhecido.

    Opcionalmente aceita `user_id` (form field) para vincular a imagem ao usuário.
    """
    contents = await arquivo.read()
    saved = save_known_face(contents, arquivo.filename or "unknown.jpg")

    # Se salvou com sucesso e um user_id foi fornecido, associe a face ao usuário
    if saved.get('ok') and user_id:
        user = users.add_face_to_user(user_id, saved.get('path'))
        if not user:
            # usuário não encontrado — retornar aviso, mas deixar o arquivo salvo
            return {**saved, 'warning': 'Usuário não encontrado para associação'}
        return {**saved, 'user': user}

    return saved


@app.post("/cadastrar_video")
async def cadastrar_video(
    arquivo: UploadFile = File(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
):
    """Recebe um video e cadastra varias imagens de referencia."""
    contents = await arquivo.read()
    saved = save_faces_from_video(contents, user_id, user_name, original_filename=arquivo.filename)

    if saved.get("ok") and user_id:
        user = users.get_user(user_id)
        if not user:
            return {**saved, "warning": "Usuario nao encontrado para associacao"}

        for path in saved.get("paths", []):
            users.add_face_to_user(user_id, os.path.basename(path))

        user = users.get_user(user_id)
        return {**saved, "user": user}

    return saved


# ===== Rotas de usuários (JSON temp DB) =====
@app.get("/usuarios")
def listar_usuarios():
    return users.list_users()


@app.get("/usuarios/{user_id}")
def obter_usuario(user_id: str):
    u = users.get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return u


@app.post("/usuarios")
def criar_usuario(payload: dict):
    # validação mínima: precisa de 'nome'
    if 'nome' not in payload:
        raise HTTPException(status_code=400, detail="'nome' é obrigatório")
    created = users.create_user(payload)
    return created


@app.put("/usuarios/{user_id}")
def atualizar_usuario(user_id: str, payload: dict):
    updated = users.update_user(user_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return updated


@app.delete("/usuarios/{user_id}")
def remover_usuario(user_id: str):
    ok = users.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return {"ok": True}
