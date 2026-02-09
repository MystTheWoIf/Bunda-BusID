import json
import os
import threading
from typing import Optional, Dict, Any, List

# caminho para o arquivo JSON (relativo ao módulo)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'users.json'))

# pequena trava para escrita segura
_lock = threading.Lock()


def _load_all() -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _save_all(users: List[Dict[str, Any]]):
    with _lock:
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)


def list_users() -> List[Dict[str, Any]]:
    return _load_all()


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    users = _load_all()
    for u in users:
        if str(u.get('id')) == str(user_id):
            return u
    return None


def create_user(user: Dict[str, Any]) -> Dict[str, Any]:
    users = _load_all()
    # gerar id simples (incremental)
    next_id = 1
    if users:
        try:
            next_id = max(int(u.get('id', 0)) for u in users) + 1
        except Exception:
            next_id = len(users) + 1

    user['id'] = next_id
    users.append(user)
    _save_all(users)
    return user


def update_user(user_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    users = _load_all()
    for i, u in enumerate(users):
        if str(u.get('id')) == str(user_id):
            u.update(data)
            users[i] = u
            _save_all(users)
            return u
    return None


def delete_user(user_id: str) -> bool:
    users = _load_all()
    for i, u in enumerate(users):
        if str(u.get('id')) == str(user_id):
            users.pop(i)
            _save_all(users)
            return True
    return False


def add_face_to_user(user_id: str, face_path: str) -> Optional[Dict[str, Any]]:
    """Anexa um caminho de imagem (face) ao usuário especificado.

    O que é armazenado no JSON é apenas o nome do arquivo (basename) para portabilidade.
    """
    users = _load_all()
    for i, u in enumerate(users):
        if str(u.get('id')) == str(user_id):
            faces = u.get('faces') or []
            # armazenar apenas basename
            face_name = os.path.basename(face_path)
            faces.append(face_name)
            u['faces'] = faces
            users[i] = u
            _save_all(users)
            return u
    return None
