import cv2
import os
import numpy as np
import tempfile
from typing import Dict, List, Optional, Tuple

# ===== Global settings =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Use a folder relative to this file so the module works regardless of current working dir
faces_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "face_Api.faces"))

known_faces: List[np.ndarray] = []
classNames: List[Dict[str, str]] = []

FACE_SIZE = (100, 100)
SIMILARITY_THRESHOLD = 0.10
BLUR_THRESHOLD = 60.0
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 215

POSES = ["frente", "cima", "baixo", "esquerda", "direita"]


def _ensure_faces_dir():
    if not os.path.exists(faces_path):
        os.makedirs(faces_path, exist_ok=True)


def _select_largest_face(faces: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _align_face_by_eyes(face_gray: np.ndarray) -> np.ndarray:
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
    if len(eyes) < 2:
        return face_gray

    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    c1 = (x1 + w1 // 2, y1 + h1 // 2)
    c2 = (x2 + w2 // 2, y2 + h2 // 2)
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (face_gray.shape[1] // 2, face_gray.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(
        face_gray,
        matrix,
        (face_gray.shape[1], face_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_face_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    center = (gray.shape[1] // 2, gray.shape[0] // 2)
    axes = (gray.shape[1] // 3, int(gray.shape[0] * 0.5))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.bitwise_and(gray, gray, mask=mask)


def _is_face_quality_ok(gray: np.ndarray) -> bool:
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < BLUR_THRESHOLD:
        return False
    brightness = float(np.mean(gray))
    if brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
        return False
    return True


def _extract_face_gray_from_gray(
    gray: np.ndarray, rect: Tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = rect
    face_gray = gray[y : y + h, x : x + w]
    face_gray = _align_face_by_eyes(face_gray)
    return cv2.resize(face_gray, FACE_SIZE)


def _extract_face_gray(frame_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _extract_face_gray_from_gray(gray, rect)


def _preprocess_face_gray(gray: np.ndarray) -> np.ndarray:
    face_gray = _apply_clahe(gray)
    face_gray = cv2.normalize(face_gray, None, 0, 255, cv2.NORM_MINMAX)
    return face_gray


def _slugify(value: str) -> str:
    keep = []
    for ch in value.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_"):
            keep.append("-")
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "user"


def load_known_faces():
    """Load all known faces from the faces folder."""
    global known_faces, classNames
    known_faces = []
    classNames = []

    _ensure_faces_dir()

    for filename in os.listdir(faces_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            stem = os.path.splitext(filename)[0]
            parts = stem.split("_", 1)
            user_id = parts[0] if parts else stem
            name_part = parts[1] if len(parts) > 1 else stem
            user_name = name_part.split("__", 1)[0]

            path = os.path.join(faces_path, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_rect = _select_largest_face(faces)

            try:
                if face_rect is not None:
                    raw_face = _extract_face_gray_from_gray(gray, face_rect)
                else:
                    raw_face = cv2.resize(gray, FACE_SIZE)
                face_img = _apply_face_mask(_preprocess_face_gray(raw_face))
            except Exception:
                continue

            known_faces.append(face_img)
            classNames.append({"id": user_id, "nome": user_name})

    print(f"[OK] {len(known_faces)} rostos carregados: {classNames}")


def recognize_face_from_image(image_bytes):
    """Used by the API - recognize face from image bytes."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    return recognize_faces_in_frame(img)


def recognize_faces_in_frame(frame):
    """Used both by the API and live mode."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for (x, y, w, h) in faces:
        try:
            raw_face = _extract_face_gray_from_gray(gray, (x, y, w, h))
            qualidade_ok = _is_face_quality_ok(raw_face)
            face_img = _apply_face_mask(_preprocess_face_gray(raw_face))
        except Exception:
            continue

        name = "Desconhecido"
        best_similarity = 1.0

        if qualidade_ok:
            for i, known_face in enumerate(known_faces):
                if known_face.shape != face_img.shape:
                    continue
                diff = cv2.absdiff(face_img, known_face)
                similarity = float(np.sum(diff)) / (FACE_SIZE[0] * FACE_SIZE[1] * 255)
                if similarity < best_similarity:
                    best_similarity = similarity
                    if similarity < SIMILARITY_THRESHOLD:
                        name = classNames[i]
        else:
            best_similarity = None

        results.append(
            {
                "coords": (int(x), int(y), int(w), int(h)),
                "similaridade": round(float(best_similarity), 4)
                if best_similarity is not None
                else None,
                "qualidade_ok": qualidade_ok,
                "id": name.get("id") if isinstance(name, dict) else None,
                "nome": name.get("nome") if isinstance(name, dict) else name,
            }
        )

    return results


def save_known_face(image_bytes: bytes, filename: str):
    """Save a new image to the known faces folder and reload."""
    _ensure_faces_dir()

    safe_name = os.path.basename(filename)
    safe_name = safe_name.replace(" ", "_")
    dest_path = os.path.join(faces_path, safe_name)

    try:
        with open(dest_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    load_known_faces()

    return {"ok": True, "path": dest_path}


def _write_temp_video(video_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        return tmp.name


def _get_video_suffix(original_filename: Optional[str]) -> str:
    if not original_filename:
        return ".webm"
    ext = os.path.splitext(original_filename)[1].lower()
    if ext in (".webm", ".mp4", ".mov", ".avi", ".mkv"): 
        return ext
    return ".webm"


def save_faces_from_video(
    video_bytes: bytes,
    user_id: str,
    user_name: str,
    original_filename: Optional[str] = None,
    seconds_per_pose: int = 3,
    frame_stride: int = 3,
) -> Dict[str, object]:
    """Extract frames from a video and save multiple faces as references."""
    _ensure_faces_dir()

    slug_name = _slugify(user_name)
    saved_paths: List[str] = []
    per_pose: Dict[str, int] = {pose: 0 for pose in POSES}

    temp_path = None
    cap = None
    try:
        suffix = _get_video_suffix(original_filename)
        temp_path = _write_temp_video(video_bytes, suffix)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened() and suffix != ".mp4":
            cap.release()
            os.remove(temp_path)
            temp_path = _write_temp_video(video_bytes, ".mp4")
            cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return {"ok": False, "error": "Video decode failed"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        frames_per_pose = int(seconds_per_pose * fps)
        pose_index = 0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok or pose_index >= len(POSES):
                break

            if frame_index % frame_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_rect = _select_largest_face(faces)
                if face_rect is not None:
                    raw_face = _extract_face_gray_from_gray(gray, face_rect)
                    if _is_face_quality_ok(raw_face):
                        face_img = _apply_face_mask(_preprocess_face_gray(raw_face))
                        pose = POSES[pose_index]
                        filename = f"{user_id}_{slug_name}__{pose}_{per_pose[pose]}.jpg"
                        path = os.path.join(faces_path, filename)
                        if cv2.imwrite(path, face_img):
                            saved_paths.append(path)
                            per_pose[pose] += 1

            frame_index += 1
            if frame_index > 0 and frame_index % frames_per_pose == 0:
                pose_index += 1

        cap.release()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    load_known_faces()
    if not saved_paths:
        return {"ok": False, "error": "No frames saved", "por_pose": per_pose, "total": 0}
    return {
        "ok": True,
        "paths": saved_paths,
        "por_pose": per_pose,
        "total": len(saved_paths),
    }
