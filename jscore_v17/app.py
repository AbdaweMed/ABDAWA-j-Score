import os, datetime, csv, io, sqlite3, math, logging, time
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict
from flask import Flask, render_template, request, flash, Response, send_from_directory, jsonify, url_for, redirect
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from dotenv import load_dotenv

# === OpenCV (optionnel) : détection de l'œil + % jaune ===
try:
    import cv2
except Exception:
    cv2 = None

def opencv_eye_yellow(image_path):
    """Retourne (eyes_detected: bool, yellow_pct: float|None)."""
    if cv2 is None:
        return False, None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hc = getattr(cv2.data, "haarcascades", None)
        if not hc:
            return False, None
        face_c = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_frontalface_default.xml"))
        eye_c  = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_eye.xml"))
        if face_c.empty() or eye_c.empty():
            return False, None
        faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        rois = []
        for (x,y,w,h) in faces:
            roi_g = gray[y:y+h, x:x+w]
            eyes = eye_c.detectMultiScale(roi_g, 1.15, 7, minSize=(20,20))
            for (ex,ey,ew,eh) in eyes:
                if ey + eh/2 < h * 0.7:
                    rois.append((x+ex, y+ey, ew, eh))
        if not rois:
            return False, None
        total = 0; yellow = 0
        for (ex,ey,ew,eh) in rois:
            patch = img[ey:ey+eh, ex:ex+ew]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            lower = (15,30,80); upper=(35,255,255)
            mask = cv2.inRange(hsv, lower, upper)
            yellow += int(mask.sum()//255)
            total  += mask.size
        if total == 0:
            return False, None
        pct = max(0.0, min(100.0, (yellow/float(total))*100.0))
        return True, round(pct, 2)
    except Exception:
        return False, None


VERSION = "15.0"
CONSENT_REF = "Ministère de la Santé MR – réf. 012-2025 (07/07/2025)"
CONSENT_VERSION = "v1.0-2025-07-07"

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER   = os.path.join(BASE_DIR, "data")
DB_PATH       = os.path.join(DATA_FOLDER, "collecte.db")
# Dynamically allow WEBP only if Pillow supports it
try:
    from PIL import features as PIL_features  # type: ignore
    _HAS_WEBP = bool(PIL_features.check("webp"))
except Exception:
    _HAS_WEBP = False
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"} | ({"webp"} if _HAS_WEBP else set())
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

# ===== Rate limiting (naïf en mémoire) =====
RATE_LIMIT = defaultdict(lambda: deque())
LIMIT_PER_MIN = 10
LIMIT_PER_HOUR = 30

def is_rate_limited(ip: str) -> bool:
    now = time.time()
    dq = RATE_LIMIT[ip]
    # purge > 1h
    while dq and now - dq[0] > 3600:
        dq.popleft()
    last_minute = [t for t in dq if now - t <= 60]
    if len(last_minute) >= LIMIT_PER_MIN or len(dq) >= LIMIT_PER_HOUR:
        return True
    dq.append(now)
    return False

load_dotenv()
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin")
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-prod")

# ===== Logging =====
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.access(UPLOAD_FOLDER, os.W_OK):
    app.logger.warning("UPLOAD_FOLDER not writable: %s", UPLOAD_FOLDER)
log_path = os.path.join(DATA_FOLDER, "app.log")
handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s"))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# ===== DB init & migration =====
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT,
            date_photo TEXT,
            date_biomarkers TEXT,
            patient_id TEXT,
            diag TEXT,
            photo_filename TEXT,
            age REAL, alt REAL, ast REAL, plt REAL, afp REAL, ggt REAL, pa REAL, bt REAL, bd REAL,
            uln_ast REAL,
            jaundice_k REAL,
            apri REAL,
            fib4 REAL,
            j_score REAL,
            j_interpretation TEXT,
            patient_signature TEXT,
            consent INTEGER,
            consent_ref TEXT,
            consent_version TEXT,
            submission_ip TEXT,
            user_agent TEXT,
            model_version TEXT
        )
        """
    )
    con.commit()

    # Migration: add columns if missing
    cur.execute("PRAGMA table_info(submissions)")
    cols = {r[1] for r in cur.fetchall()}
    add_cols = []
    if "consent_ref" not in cols:
        add_cols.append(("consent_ref", "TEXT"))
    if "consent_version" not in cols:
        add_cols.append(("consent_version", "TEXT"))
    if "submission_ip" not in cols:
        add_cols.append(("submission_ip", "TEXT"))
    if "user_agent" not in cols:
        add_cols.append(("user_agent", "TEXT"))
    for name, typ in add_cols:
        try:
            cur.execute(f"ALTER TABLE submissions ADD COLUMN {name} {typ}")
        except Exception:
            pass
    con.commit(); con.close()

init_db()

# ===== Utils =====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None

def to_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def generate_patient_id():
    import random, string
    return datetime.datetime.utcnow().strftime("P-%Y%m%d-") + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

def require_basic_auth(auth_header):
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    try:
        import base64 as b64
        user, pwd = b64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8").split(":", 1)
        return user == ADMIN_USER and pwd == ADMIN_PASS
    except Exception:
        return False

# ===== Scores & interpretation =====
def compute_apri(ast, uln_ast, plt):
    if ast is None or plt in (None, 0):
        return None
    if not uln_ast:
        uln_ast = 40.0
    return round(((ast / uln_ast) * 100.0) / plt, 4)

def compute_fib4(age, ast, alt, plt):
    if None in (age, ast, alt, plt) or plt == 0 or alt <= 0:
        return None
    return round((age * ast) / (plt * math.sqrt(alt)), 4)

def classify_apri(apri):
    if apri is None: return None
    if apri < 0.5: return "normal"
    elif apri >= 2.0: return "cirrhose"
    else: return "fibrose"

def classify_fib4(fib4):
    if fib4 is None: return None
    if fib4 < 1.45: return "normal"
    elif fib4 >= 3.25: return "cirrhose"
    else: return "fibrose"

def is_bt_normal(bt):
    try:
        b = float(bt)
        return 3.0 <= b <= 12.0
    except Exception:
        return False

def bucket_k(k):
    if k is None: return 1.5
    choices = [1.0, 1.5, 2.0]
    return min(choices, key=lambda c: abs(c - float(k)))

def interpret_j_score(apri, fib4, bt, k):
    ca, cf = classify_apri(apri), classify_fib4(fib4)
    bt_ok, k_b = is_bt_normal(bt), bucket_k(k)
    if ca == "cirrhose" and cf == "cirrhose" and not bt_ok:
        return "Urgemment consulter un médecin / استشارة طبيب بشكل عاجل"
    if ca == "fibrose" and cf == "fibrose" and not bt_ok:
        return "Consulter un médecin / استشارة طبيب"
    if ca == "normal" and cf == "normal" and bt_ok:
        return "Consulter un médecin / استشارة طبيب" if abs(k_b - 2.0) < 1e-6 else "Normale / طبيعي"
    if (ca in ("fibrose", "cirrhose")) or (cf in ("fibrose", "cirrhose")):
        return "Consulter un médecin / استشارة طبيب"
    return "À évaluer par un médecin / يحتاج إلى تقييم طبي"

# ===== K (focalisé périlimbique) & qualité photo =====
def assess_photo_quality(image_path):
    im = Image.open(image_path).convert("L").resize((256, 256))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    mean, std = float(arr.mean()), float(arr.std())
    too_dark = mean < 0.2
    too_bright = mean > 0.9
    low_contrast = std < 0.08
    highlight_frac = float((arr > 0.98).mean())
    reflections = highlight_frac > 0.02
    return {"too_dark": too_dark, "too_bright": too_bright, "low_contrast": low_contrast, "reflections": reflections}


def rough_yellow_ratio(image_path: str) -> float:
    """Very simple heuristic: fraction of pixels in the central area that look 'yellow' in HSV.
    This is NOT a diagnosis; it's a coarse pre-signal used when precise K estimation fails."""
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            # Crop central area (60% of width/height)
            cw, ch = int(w*0.6), int(h*0.6)
            x0, y0 = (w - cw)//2, (h - ch)//2
            im = im.crop((x0, y0, x0+cw, y0+ch))
            arr = np.asarray(im).astype("float32") / 255.0
            # Convert to HSV
            r, g, b = arr[...,0], arr[...,1], arr[...,2]
            cmax = arr.max(axis=-1); cmin = arr.min(axis=-1); delta = cmax - cmin + 1e-6
            # Hue in degrees
            h_deg = np.where(cmax==r, (60*((g-b)/delta)%360),
                     np.where(cmax==g, 60*(((b-r)/delta)+2), 60*(((r-g)/delta)+3)))
            s = np.where(cmax==0, 0, delta/cmax)
            v = cmax
            yellow_mask = (h_deg >= 35.0) & (h_deg <= 65.0) & (s >= 0.20) & (v >= 0.50)
            return float(yellow_mask.mean())
    except Exception:
        return 0.0


def compute_jaundice_k_focused(image_path, debug_overlay=False):
    MAX_SIDE = 640; BLUR_RAD = 3; ANGLES = 72
    R_MIN_FRAC, R_MAX_FRAC = 0.06, 0.30; R_STEP = 1.0
    RING_OFFSET = 4; RING_THICK = 10
    S_MIN = 0.20; V_MIN = 0.50; H_Y_MIN, H_Y_MAX = 35.0, 65.0
    K_MIN, K_MAX = 1.0, 3.0
    im = Image.open(image_path); im = ImageOps.exif_transpose(im).convert("RGB")
    w0, h0 = im.size
    if max(w0, h0) > MAX_SIDE:
        if w0 >= h0:
            new_w, new_h = MAX_SIDE, int(round(h0 * (MAX_SIDE / w0)))
        else:
            new_h, new_w = MAX_SIDE, int(round(w0 * (MAX_SIDE / h0)))
        im_small = im.resize((new_w, new_h), Image.LANCZOS)
    else:
        im_small = im.copy()
    w, h = im_small.size
    hsv = im_small.convert("HSV"); H, S, V = [np.asarray(ch, dtype=np.float32) for ch in hsv.split()]
    H_deg = H * (360.0 / 255.0); S_n = S / 255.0; V_n = V / 255.0
    V_blur = np.asarray(im_small.filter(ImageFilter.GaussianBlur(radius=BLUR_RAD)).convert("L"), dtype=np.float32)
    x0, x1 = int(0.2 * w), int(0.8 * w); y0, y1 = int(0.2 * h), int(0.8 * h)
    sub = V_blur[y0:y1, x0:x1]
    iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
    cy, cx = iy + y0, ix + x0

    thetas = np.linspace(0, 2 * np.pi, ANGLES, endpoint=False)

    def sample(arr, x, y):
        x = min(max(x, 0), w - 1); y = min(max(y, 0), h - 1)
        x0 = int(np.floor(x)); x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y)); y1 = min(y0 + 1, h - 1)
        dx = x - x0; dy = y - y0
        v00 = arr[y0, x0]; v10 = arr[y0, x1]; v01 = arr[y1, x0]; v11 = arr[y1, x1]
        return v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy

    r_min = max(8, int(min(w, h) * R_MIN_FRAC)); r_max = int(min(w, h) * R_MAX_FRAC)
    radii = np.arange(r_min, r_max, R_STEP, dtype=np.float32)
    ring_mean = []
    for r in radii:
        vals = [sample(V_n, cx + r * np.cos(t), cy + r * np.sin(t)) for t in thetas]
        ring_mean.append(np.mean(vals))
    ring_mean = np.asarray(ring_mean, dtype=np.float32)
    grad = np.zeros_like(ring_mean); grad[1:-1] = ring_mean[2:] - ring_mean[:-2]
    best_idx = int(np.argmax(grad)); r_iris = float(radii[best_idx])

    yy, xx = np.mgrid[0:h, 0:w]
    R = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = (R >= (r_iris + RING_OFFSET)) & (R <= (r_iris + RING_OFFSET + RING_THICK))
    ring_mask &= (V_n >= V_MIN)
    valid = ring_mask & (S_n >= S_MIN)
    total = int(np.count_nonzero(valid))
    if total < 50:
        valid = (V_n >= V_MIN) & (S_n >= S_MIN)
        total = int(np.count_nonzero(valid))
    if total == 0:
        fraction_yellow = 0.0
    else:
        yellow = valid & (H_deg >= H_Y_MIN) & (H_deg <= H_Y_MAX)
        n_yellow = int(np.count_nonzero(yellow))
        fraction_yellow = n_yellow / float(total)
    k = float(np.clip(1.0 + 2.0 * fraction_yellow, K_MIN, K_MAX))
    return k, {"fraction_yellow": fraction_yellow}

# ===== PDF =====
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def generate_pdf(submission_id, data):
    pdf_dir = os.path.join(DATA_FOLDER, "pdfs"); os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"submission_{submission_id}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Header: logo + centered title + divider
    try:
        logo_path = os.path.join(BASE_DIR, 'static', 'logo_abdawa.png')
        if os.path.exists(logo_path):
            c.drawImage(logo_path, 1.5*cm, h - 3.0*cm, width=1.8*cm, height=1.8*cm, mask='auto')
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w/2, h - 2.0*cm, "ABDAWA J‑Score – Récapitulatif")
    c.setLineWidth(0.7); c.line(1.5*cm, h - 2.2*cm, w - 1.5*cm, h - 2.2*cm)

    y = h - 3.2 * cm
    def line(t, bold=False):
        nonlocal y
        if bold:
            c.setFont("Helvetica-Bold", 11)
        else:
            c.setFont("Helvetica", 11)
        c.drawString(2 * cm, y, str(t)); y -= 0.8 * cm

    # Body
    line(f"ID: {data.get('patient_id','')}    Photo: {data.get('date_photo','')}    Bio: {data.get('date_biomarkers','')}", bold=True)
    line(f"Diag: {data.get('diag','')}")
    jrate = data.get('yellow_ratio', 0.0)
    try: jrate_pct = round(max(0.0, min(1.0, float(jrate))) * 100.0, 1)
    except Exception: jrate_pct = jrate
    line(f"BT: {data.get('bt')}    K: {data.get('k')}    Taux jaunissement détecté: {jrate_pct} %")
    line(f"J-score: {data.get('j_score')}    Interprétation: {data.get('interp')}", bold=True)

    y -= 0.4 * cm
    c.setFont("Helvetica-Bold", 12); line("Consentement éclairé / الموافقة المستنيرة", bold=True)
    c.setFont("Helvetica", 10)
    for t in [
        "Étude autorisée par le Ministère de la Santé de la Mauritanie.",
        f"Référence éthique: {CONSENT_REF} (version {CONSENT_VERSION}).",
        "Je consens à l'utilisation de mes données (photo de l’œil et biomarqueurs) à des fins de recherche et développement.",
        "أوافق على استخدام بياناتي لأغراض البحث والتطوير. تمت الموافقة الأخلاقية من وزارة الصحة الموريتانية.",
    ]:
        line(t)
    y -= 0.4 * cm
    line(f"Signature: {data.get('patient_signature','')}    Consentement: {'Oui' if data.get('consent') else 'Non'}")
    c.showPage(); c.save()
    return pdf_path


# ===== Routes =====
@app.route("/", methods=["GET"])
def index():
    return render_template("form.html", result_score=None, result_interp=None, k_value=None, yellow_ratio=None, today=datetime.date.today().isoformat(), version=VERSION)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", version=VERSION, consent_ref=CONSENT_REF, consent_version=CONSENT_VERSION)

@app.route("/health", methods=["GET"])
def health():
    try:
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM submissions")
        count = int(cur.fetchone()[0]); con.close()
        return jsonify({"status": "ok", "version": VERSION, "db_submissions": count}), 200
    except Exception as e:
        app.logger.exception("health failed")
        return jsonify({"status": "error", "version": VERSION, "error": str(e)}), 500

@app.route("/submit", methods=["POST"])
def submit():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    ua = request.headers.get("User-Agent", "unknown")

    # Rate limit
    if is_rate_limited(ip):
        flash("Limite de soumissions atteinte. Réessayez plus tard.", "error")
        return render_template("form.html", result_score=None, result_interp=None, k_value=None, yellow_ratio=None, today=datetime.date.today().isoformat(), version=VERSION)

    date_photo_str = request.form.get("date_photo") or ""
    date_bio_str   = request.form.get("date_bio") or ""

    patient_id = (request.form.get("patient_id") or "").strip() or generate_patient_id()
    diag = request.form.get("diag") or ""
    patient_signature = (request.form.get("patient_signature") or "").strip()
    consent = request.form.get("consent") == "on"

    age = to_float(request.form.get("age")); alt = to_float(request.form.get("alt"))
    ast = to_float(request.form.get("ast")); plt = to_float(request.form.get("plt"))
    afp = to_float(request.form.get("afp")); ggt = to_float(request.form.get("ggt")); pa  = to_float(request.form.get("pa"))
    bt  = to_float(request.form.get("bt"));  bd  = to_float(request.form.get("bd"))
    uln_ast = to_float(request.form.get("uln_ast")) or 40.0

    errs = []
    if not to_date(date_photo_str): errs.append("Date de la photo invalide.")
    if not to_date(date_bio_str):   errs.append("Date des biomarqueurs invalide.")
    file = request.files.get("eye_photo")
    if not file or file.filename == "" or not allowed_file(file.filename): errs.append("Photo manquante ou extension non autorisée.")
    if errs:
        for e in errs: flash(e, "error")
        return render_template("form.html", result_score=None, result_interp=None, k_value=None, yellow_ratio=None, today=datetime.date.today().isoformat(), version=VERSION)

    fname = secure_filename(file.filename)
    timestamp_utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stored = f"{timestamp_utc}_{os.path.splitext(fname)[0]}{os.path.splitext(fname)[1].lower()}"
    img_path = os.path.join(UPLOAD_FOLDER, stored)
    file.save(img_path)

    # Validate that the uploaded file is a real image
    try:
        with Image.open(img_path) as _im:
            _im.verify()
    except Exception:
        app.logger.exception("Uploaded file is not a valid image")
        try:
            os.remove(img_path)
        except Exception:
            pass
        flash("Le fichier envoyé n'est pas une image valide ou le format n'est pas supporté (PNG/JPG/JPEG{webp}).".format(webp="/WEBP" if "webp" in ALLOWED_EXTENSIONS else ""), "error")
        return render_template("form.html",
                               result_score=None, result_interp=None, k_value=None, yellow_ratio=None,
                               today=datetime.date.today().isoformat(), version=VERSION)
    q = assess_photo_quality(img_path)
    if q["too_dark"]:     flash("Recommandé de reprendre ou retéléverser la photo.", "warning")
    if q["too_bright"]:   flash("Recommandé de reprendre ou retéléverser la photo.", "warning")
    if q["low_contrast"]: flash("Recommandé de reprendre ou retéléverser la photo.", "warning")
    if q["reflections"]:  flash("Recommandé de reprendre ou retéléverser la photo.", "warning")
    try:
        k, kd = compute_jaundice_k_focused(img_path, debug_overlay=os.environ.get("DEBUG_K","0")=="1")
    except Exception:
        app.logger.exception("compute_jaundice_k_focused failed")
        # Fallback: compute a rough yellow ratio and continue (no hard failure)
        k, kd = None, {"fraction_yellow": rough_yellow_ratio(img_path)}
        flash("Recommandé de reprendre ou retéléverser la photo.", "warning")
    apri = compute_apri(ast, uln_ast, plt)
    fib4 = compute_fib4(age, ast, alt, plt)

    # === OpenCV eye check (optionnel) ===
    cv_eyes_detected, cv_yellow_pct = opencv_eye_yellow(img_path)
    k_for_score = 1.0 if (not cv_eyes_detected or cv_yellow_pct is None) else k

    j_score = None
    if apri is not None and fib4 not in (None, 0) and bt is not None and k_for_score not in (None, 0):
        j_score = round((apri / fib4) * (bt * k_for_score), 4)
    interp = interpret_j_score(apri, fib4, bt, k_for_score)
    if k_for_score == 1.0 and (not cv_eyes_detected or cv_yellow_pct is None):
        interp = (interp + " Recommandé de reprendre ou retéléverser la photo.")
        flash("Recommandé de reprendre ou retéléverser la photo.", "warning")

    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute(
        """
        INSERT INTO submissions
        (timestamp_utc, date_photo, date_biomarkers, patient_id, diag, photo_filename,
         age, alt, ast, plt, afp, ggt, pa, bt, bd, uln_ast, jaundice_k, apri, fib4, j_score, j_interpretation,
         patient_signature, consent, consent_ref, consent_version, submission_ip, user_agent, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp_utc, date_photo_str, date_bio_str, patient_id, diag, stored,
         age, alt, ast, plt, afp, ggt, pa, bt, bd, uln_ast, k, apri, fib4, j_score, interp,
         patient_signature, int(consent), CONSENT_REF, CONSENT_VERSION, ip, ua, VERSION)
    )
    sub_id = cur.lastrowid; con.commit(); con.close()

    app.logger.info(f"New submission id={sub_id} pid={patient_id} ip={ip} ua={ua} diag={diag} j={j_score} i={interp}")

    try:
        pdf_path = generate_pdf(sub_id, {
            "patient_id": patient_id, "date_photo": date_photo_str, "date_biomarkers": date_bio_str,
            "diag": diag, "bt": bt, "k": k, "yellow_ratio": kd.get("fraction_yellow", 0.0),
            "j_score": j_score, "interp": interp,
            "patient_signature": patient_signature, "consent": consent
        })
        pdf_name = os.path.basename(pdf_path)
        flash(f'PDF récapitulatif : <a href="{url_for("serve_pdf", filename=pdf_name)}" target="_blank">télécharger</a>', "success")
    except Exception as e:
        app.logger.exception("pdf failed")
        flash("PDF non généré (voir logs).", "error")

    flash("Soumission enregistrée avec succès.", "success")
    if j_score is not None:
        flash(f"J-score = {j_score}", "info")
        flash(f"Interprétation : {interp}", "info")
    else:
        flash("J-score non calculé (données insuffisantes).", "info")
    return render_template("form.html", result_score=j_score, result_interp=interp, k_value=k, yellow_ratio=kd.get("fraction_yellow", 0.0), today=datetime.date.today().isoformat(), version=VERSION)

# ===== Static serving =====
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Admin\""})
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(os.path.join(DATA_FOLDER, "pdfs"), filename, as_attachment=True)

# ===== Admin & export =====
@app.route("/admin", methods=["GET"])
def admin_list():
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Admin\""})
    q = (request.args.get("q") or "").strip()
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    if q:
        like = f"%{q}%"
        cur.execute(
            """SELECT id, timestamp_utc, patient_id, j_score, j_interpretation, photo_filename
               FROM submissions WHERE patient_id LIKE ? OR diag LIKE ?
               ORDER BY id DESC LIMIT 200""",
            (like, like),
        )
    else:
        cur.execute(
            """SELECT id, timestamp_utc, patient_id, j_score, j_interpretation, photo_filename
               FROM submissions ORDER BY id DESC LIMIT 200"""
        )
    rows = cur.fetchall(); con.close()
    return render_template("admin.html", rows=rows, q=q, version=VERSION)

@app.route("/admin/<int:sub_id>.json", methods=["GET"])
def admin_detail_json(sub_id):
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Admin\""})
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute(
        """SELECT id, timestamp_utc, date_photo, date_biomarkers, patient_id, diag, photo_filename,
                         age, alt, ast, plt, afp, ggt, pa, bt, bd, uln_ast, jaundice_k, apri, fib4,
                         j_score, j_interpretation, patient_signature, consent, consent_ref, consent_version,
                         submission_ip, user_agent, model_version
           FROM submissions WHERE id=?""",
        (sub_id,),
    )
    row = cur.fetchone(); con.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    keys = ["id","timestamp_utc","date_photo","date_biomarkers","patient_id","diag","photo_filename",
            "age","alt","ast","plt","afp","ggt","pa","bt","bd","uln_ast","jaundice_k","apri","fib4",
            "j_score","j_interpretation","patient_signature","consent","consent_ref","consent_version",
            "submission_ip","user_agent","model_version"]
    return jsonify(dict(zip(keys, row)))

@app.route("/admin/delete/<int:sub_id>", methods=["POST","GET"])
def admin_delete(sub_id):
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Admin\""})
    confirm = request.args.get("confirm") == "1" or request.form.get("confirm") == "1"
    if not confirm:
        return Response("Ajouter ?confirm=1 pour confirmer la suppression.", 400)
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    # fetch filename to delete image
    cur.execute("SELECT photo_filename FROM submissions WHERE id=?", (sub_id,))
    r = cur.fetchone()
    cur.execute("DELETE FROM submissions WHERE id=?", (sub_id,))
    con.commit(); con.close()
    if r and r[0]:
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, r[0]))
        except Exception:
            pass
    return redirect(url_for("admin_list"))

@app.route("/export/csv", methods=["GET"])
def export_csv():
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Export CSV\""})
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute(
        """SELECT timestamp_utc, date_photo, date_biomarkers, patient_id, diag, photo_filename,
                         age, alt, ast, plt, afp, ggt, pa, bt, bd, uln_ast, jaundice_k, apri, fib4, j_score,
                         j_interpretation, patient_signature, consent, consent_ref, consent_version, submission_ip,
                         user_agent, model_version
           FROM submissions ORDER BY id DESC"""
    )
    rows = cur.fetchall(); con.close()
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(["timestamp_utc","date_photo","date_biomarkers","patient_id","diag","photo_filename",
                     "age","alt","ast","plt","afp","ggt","pa","bt","bd","uln_ast","jaundice_k","apri","fib4",
                     "j_score","j_interpretation","patient_signature","consent","consent_ref","consent_version",
                     "submission_ip","user_agent","model_version"])
    for r in rows:
        writer.writerow(r)
    return Response(output.getvalue(), 200, {"Content-Type": "text/csv; charset=utf-8",
                                             "Content-Disposition": "attachment; filename=export.csv"})


@app.route("/admin/print/<int:sub_id>", methods=["GET"])
def admin_print(sub_id):
    auth = request.headers.get("Authorization", "")
    if not require_basic_auth(auth):
        return Response("Authentification requise", 401, {"WWW-Authenticate": "Basic realm=\"Admin\""})
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute(
        """SELECT id, timestamp_utc, date_photo, date_biomarkers, patient_id, diag, photo_filename,
                         age, alt, ast, plt, afp, ggt, pa, bt, bd, uln_ast, jaundice_k, apri, fib4,
                         j_score, j_interpretation, patient_signature, consent, consent_ref, consent_version,
                         submission_ip, user_agent, model_version
           FROM submissions WHERE id=?""",
        (sub_id,),
    )
    row = cur.fetchone(); con.close()
    if not row:
        return Response("Non trouvé", 404)
    keys = ["id","timestamp_utc","date_photo","date_biomarkers","patient_id","diag","photo_filename",
            "age","alt","ast","plt","afp","ggt","pa","bt","bd","uln_ast","jaundice_k","apri","fib4",
            "j_score","j_interpretation","patient_signature","consent","consent_ref","consent_version",
            "submission_ip","user_agent","model_version"]
    data = dict(zip(keys, row))
    # Approximate yellow percent from K (k = 1 + 2*f)
    try:
        k = float(data.get("jaundice_k") or 0)
        frac = max(0.0, min(1.0, (k - 1.0) / 2.0))
        data["yellow_pct"] = round(frac * 100.0, 1)
    except Exception:
        data["yellow_pct"] = None
    # Build expected pdf filename
    pdf_name = f"submission_{sub_id}.pdf"
    return render_template("print.html", d=data, pdf_name=pdf_name)



@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    flash("Fichier trop volumineux (max 10 Mo).", "error")
    return render_template("form.html",
                           result_score=None, result_interp=None, k_value=None, yellow_ratio=None,
                           today=datetime.date.today().isoformat(), version=VERSION), 413

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("Unhandled 500")
    flash("Une erreur interne est survenue pendant le traitement de la photo.", "error")
    return render_template("form.html",
                           result_score=None, result_interp=None, k_value=None, yellow_ratio=None,
                           today=datetime.date.today().isoformat(), version=VERSION), 500

if __name__ == '__main__':
    app.run(debug=False)