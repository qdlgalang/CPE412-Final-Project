from pathlib import Path
from datetime import datetime
import json, re, sqlite3, os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import joblib
except ImportError:
    joblib = None

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "maritess.db"
app = Flask(__name__)
app.secret_key = os.environ.get("MARITESS_SECRET_KEY", "change-this-demo-secret-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

ROLES = {"frontdesk": "Front Desk", "vaw": "VAW Desk Officer", "secretary": "Secretary"}
DEMO_USERS = {
    "frontdesk": ("frontdesk123", "frontdesk"),
    "vaw": ("vaw123", "vaw"),
    "secretary": ("secretary123", "secretary"),
}

SERVICE_MODEL = None
URGENCY_MODEL = None
if joblib:
    try:
        SERVICE_MODEL = joblib.load(BASE / "models" / "service_model.joblib")
        URGENCY_MODEL = joblib.load(BASE / "models" / "urgency_model.joblib")
    except Exception:
        pass

SERVICES = ["Barangay Clearance", "Certificate of Residency", "Certificate of Indigency", "Business Clearance", "Lupon Complaint", "VAW Report", "Child Protection Report", "Disaster/Incident Report", "Emergency Assistance", "Incident Report"]
URGENT_PATTERNS = {
    "Abuse / VAW": [r"\b(he hit|she hit|physical abuse|sexual abuse|abuse|sinaktan|binugbog|rape|molest|ex[- ]?boyfriend threatens|threatened by partner|domestic violence|vaw)\b"],
    "Child protection": [r"\b(child abuse|battered child|child is being hurt|molested child|abandoned child|child protection)\b"],
    "Medical emergency": [r"\b(cannot breathe|not breathing|unconscious|heart attack|severe bleeding|medical emergency|needs ambulance)\b"],
    "Disaster": [r"\b(fire is spreading|building collapse|flooding now|trapped by flood|earthquake damage|house is burning|disaster happening)\b"],
    "Immediate threat / crime": [r"\b(about to kill|has a weapon|weapon right now|currently attacking|threatening to kill|crime in progress|being chased with a knife)\b"],
}

def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect_db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL)''')
        con.execute('''CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, created_by TEXT NOT NULL,
            role_scope TEXT NOT NULL DEFAULT 'general', text_input TEXT NOT NULL, service TEXT NOT NULL,
            entities_json TEXT NOT NULL, urgency TEXT NOT NULL, urgency_category TEXT, referral TEXT,
            missing_requirements TEXT NOT NULL, next_action TEXT NOT NULL, summary TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'New', staff_notes TEXT DEFAULT ''
        )''')
        for username, (password, role) in DEMO_USERS.items():
            existing = con.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
            if not existing:
                con.execute("INSERT INTO users VALUES (?, ?, ?)", (username, generate_password_hash(password), role))

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def role_required(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if session.get("role") not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def detect_urgency(text):
    lower = text.lower()
    for category, patterns in URGENT_PATTERNS.items():
        if any(re.search(p, lower, flags=re.I) for p in patterns):
            referral = "VAW Desk Officer" if category == "Abuse / VAW" else "BCPC / authorized child-protection focal person" if category == "Child protection" else "Emergency services / QC Helpline 122 as locally verified" if category in ("Medical emergency", "Disaster", "Immediate threat / crime") else "Authorized staff"
            return "Urgent", category, referral
    if URGENCY_MODEL:
        try:
            pred = str(URGENCY_MODEL.predict([text])[0]).lower()
            if pred == "urgent":
                return "Review needed", "Classifier flagged possible urgency", "Staff review required"
        except Exception:
            pass
    return "Non-Urgent", "", ""

def classify_service(text):
    if SERVICE_MODEL:
        try:
            prediction = str(SERVICE_MODEL.predict([text])[0])
            if prediction in SERVICES:
                return prediction
        except Exception:
            pass
    low = text.lower()
    mapping = [
        (["vaw", "abuse", "ex-boyfriend", "ex boyfriend", "partner hit", "sinaktan"], "VAW Report"),
        (["child abuse", "child protection", "battered child", "molested child"], "Child Protection Report"),
        (["fire", "flood", "earthquake", "disaster"], "Disaster/Incident Report"),
        (["ambulance", "unconscious", "medical emergency", "cannot breathe"], "Emergency Assistance"),
        (["lupon", "mediation", "neighbor dispute", "complaint against neighbor"], "Lupon Complaint"),
        (["business permit", "business clearance", "business permit"], "Business Clearance"),
        (["indigency", "certificate of indigency", "financial assistance"], "Certificate of Indigency"),
        (["residency", "resident certificate", "proof of residence"], "Certificate of Residency"),
        (["clearance", "police clearance", "barangay clearance"], "Barangay Clearance"),
        (["incident report", "lost item", "lost phone", "theft"], "Incident Report"),
    ]
    for terms, service in mapping:
        if any(term in low for term in terms): return service
    return "Needs staff confirmation"

def extract_entities(text):
    # Conservative baseline: do not pretend regex is a full NER model.
    name = None
    patterns = [r"(?:my name is|i am|i'm|resident named|resident)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:from|of|at|needs|is asking)"]
    for pat in patterns:
        m = re.search(pat, text.strip(), flags=re.I)
        if m:
            candidate = m.group(1).strip()
            if len(candidate.split()) >= 2 and candidate.lower() not in {"barangay clearance", "certificate of residency"}:
                name = candidate
                break
    address = None
    addr = re.search(r"\b(?:at|address is|lives at|from)\s+(#?\d+\s+[A-Za-z0-9 .'-]{3,70}(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Drive|Dr\.?|Lane|Ln\.?))", text, re.I)
    if addr: address = re.sub(r"\s+", " ", addr.group(1)).strip(" .,")
    purpose = None
    pm = re.search(r"\b(?:for|because of|purpose is)\s+([^.;,]{3,80})", text, re.I)
    if pm: purpose = pm.group(1).strip()
    return {"name": name, "address": address, "purpose": purpose, "requested_service": None}

def knowledge_answer(question):
    docs = BASE / "knowledge_docs"
    chunks = []
    for path in docs.glob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8")
            if content.strip(): chunks.append((path.name, content.strip()))
        except OSError: pass
    if not chunks:
        return {"answer": "No verified barangay documents have been added yet. I cannot confirm official requirements, fees, office hours, or procedures. Please check the current Citizen's Charter or ask the authorized barangay staff.", "source": "No verified source loaded", "date": "Not available"}
    qwords = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    scored = []
    for filename, content in chunks:
        score = len(qwords & set(re.findall(r"[a-zA-Z]{3,}", content.lower())))
        if score: scored.append((score, filename, content))
    if not scored:
        return {"answer": "I could not find a matching answer in the loaded documents. Please confirm with authorized staff.", "source": "No matching document section", "date": "Not available"}
    scored.sort(reverse=True)
    score, filename, content = scored[0]
    # Simple document lookup only; not yet a full RAG/LLM implementation.
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    matching = [line for line in lines if set(qwords & set(re.findall(r"[a-zA-Z]{3,}", line.lower())))]
    answer = " ".join(matching[:4]) or "A related document was found; please review it with authorized staff."
    return {"answer": answer[:1200], "source": filename, "date": "Add document date in the document header"}

def make_summary(service, entities, urgency, category, referral, missing, next_action):
    return (f"Service: {service}\nResident name: {entities.get('name') or 'Not captured — staff to confirm'}\n"
            f"Address: {entities.get('address') or 'Not captured — staff to confirm'}\nPurpose: {entities.get('purpose') or 'Not captured — staff to confirm'}\n"
            f"Urgency: {urgency}{(' ('+category+')') if category else ''}\nReferral: {referral or 'No referral suggested by current rules'}\n"
            f"Requirements: {missing}\nSuggested next action: {next_action}\n\nStaff must verify all extracted details before acting.")

@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("dashboard" if session.get("username") else "login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        with connect_db() as con:
            user = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear(); session["username"] = user["username"]; session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    if request.method == "POST":
        text = request.form.get("text_input", "").strip()
        if not text:
            flash("Enter the resident's request first.", "error")
        elif len(text) > 5000:
            flash("Please limit the request to 5,000 characters.", "error")
        else:
            service = classify_service(text)
            entities = extract_entities(text); entities["requested_service"] = service
            urgency, category, referral = detect_urgency(text)
            # Official checklist intentionally disabled until approved requirements are supplied.
            missing = "Not configured — verify against the current official Citizen's Charter."
            next_action = "Staff review required" if service == "Needs staff confirmation" or urgency == "Review needed" else "Verify details and official requirements"
            scope = "vaw" if category == "Abuse / VAW" else "child" if category == "Child protection" else "general"
            summary = make_summary(service, entities, urgency, category, referral, missing, next_action)
            # VAW/child protection cases are stored with restricted scope and only shown to authorized desks.
            with connect_db() as con:
                cur = con.execute("""INSERT INTO cases (created_at,created_by,role_scope,text_input,service,entities_json,urgency,urgency_category,referral,missing_requirements,next_action,summary,status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (datetime.now().isoformat(timespec="seconds"), session["username"], scope, text, service, json.dumps(entities), urgency, category, referral, missing, next_action, summary, "New"))
                case_id = cur.lastrowid
            result = {"id": case_id, "service": service, "entities": entities, "urgency": urgency, "category": category, "referral": referral, "missing": missing, "next_action": next_action, "summary": summary}
    with connect_db() as con:
        if session.get("role") == "vaw":
            rows = con.execute("SELECT id,created_at,service,urgency,status,role_scope FROM cases ORDER BY id DESC LIMIT 50").fetchall()
        else:
            rows = con.execute("SELECT id,created_at,service,urgency,status,role_scope FROM cases WHERE role_scope='general' ORDER BY id DESC LIMIT 50").fetchall()
    return render_template("dashboard.html", result=result, cases=rows, role_name=ROLES.get(session.get("role"), session.get("role")))

@app.route("/case/<int:case_id>")
@login_required
def case_detail(case_id):
    with connect_db() as con:
        row = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not row: abort(404)
    if row["role_scope"] in ("vaw", "child") and session.get("role") != "vaw": abort(403)
    return render_template("case.html", case=row, entities=json.loads(row["entities_json"]))

@app.route("/case/<int:case_id>/update", methods=["POST"])
@login_required
def update_case(case_id):
    status = request.form.get("status", "New")
    allowed = {"New", "Waiting for Documents", "For Signature", "Completed"}
    if status not in allowed: abort(400)
    notes = request.form.get("staff_notes", "")[:2000]
    with connect_db() as con:
        row = con.execute("SELECT role_scope FROM cases WHERE id=?", (case_id,)).fetchone()
        if not row: abort(404)
        if row["role_scope"] in ("vaw", "child") and session.get("role") != "vaw": abort(403)
        con.execute("UPDATE cases SET status=?,staff_notes=? WHERE id=?", (status, notes, case_id))
    flash("Case updated.", "success")
    return redirect(url_for("case_detail", case_id=case_id))

@app.route("/knowledge", methods=["GET", "POST"])
@login_required
def knowledge():
    result = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question: result = knowledge_answer(question)
    return render_template("knowledge.html", result=result)

@app.errorhandler(403)
def forbidden(_):
    return render_template("error.html", code=403, message="You do not have permission to view this restricted case."), 403

@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="The requested case was not found."), 404

init_db()
if __name__ == "__main__":
    app.run(debug=True)
