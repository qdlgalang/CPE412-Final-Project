# MARITESS — working prototype

A local Flask prototype for the MARITESS capstone. It provides staff login, basic intake, baseline service classification, conservative extraction, urgent-language alerts, case summaries/queue, status updates, restricted VAW/child-protection cases, and a placeholder document lookup.

## 1. Requirements
- Python 3.10 or newer
- VS Code
- Internet connection only to install Python packages

## 2. Setup in VS Code (Windows PowerShell)
Open this folder in VS Code, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
python app.py
```

If PowerShell blocks virtual-environment activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open `http://127.0.0.1:5000` in your browser.

## 3. Demo logins
- Front desk: `frontdesk` / `frontdesk123`
- VAW desk officer: `vaw` / `vaw123`
- Secretary: `secretary` / `secretary123`

These are for local demonstration only. Do not deploy with these accounts or the default secret key. The prototype currently has no user-management page; add one before operational use.

## 4. Dataset
`data.csv` is the uploaded dataset. Run `python train_models.py` to create the baseline TF-IDF + Logistic Regression models under `models/` and output initial metrics. Do not interpret a random split score as real-world performance. Build an independently authored challenge set, especially typos, abbreviated requests, ambiguous cases, and urgent cases, before reporting results.

## 5. Official requirements and knowledge documents
No official barangay documents were supplied when this prototype was created. Therefore the requirements checker deliberately says `Not configured` rather than inventing a checklist. Once the barangay provides approved documents, create verified UTF-8 `.txt` copies inside `knowledge_docs/`. Include title, issue/approval date, section, and exact text. Then implement a verified service-to-requirements checklist and improve retrieval/citation testing.

## 6. Objective mapping
- **1.1:** login, role label, intake form, basic extraction.
- **1.2:** baseline service classification and a clearly unconfigured requirement-check field. Official requirement checking is NOT complete until approved checklist data is added.
- **1.3:** document lookup page; currently a simple keyword lookup, not full RAG. Verified source citation/date handling needs to be finalized with official documents.
- **1.4:** urgent phrase rules plus a model flag for review; VAW/child-protection case visibility is restricted to the VAW role. Referral text is a placeholder and must be approved by the barangay. This does not contact emergency services.
- **1.5:** generated draft summary, saved queue, editable status and notes.

## 7. Important prototype limitations
- Use synthetic test data only. Do not enter real VAW, child-protection, medical, or other sensitive case details.
- Urgency rules are not a validated emergency detector; false negatives are possible. Staff must follow approved protocols.
- Extracted names/addresses/purposes are only best-effort patterns and must be checked.
- The supplied dataset may contain synthetic or assumed service rules. Do not use its missing-requirements labels as official policy.
- Flask debug mode is enabled for local development only. Never expose this server to the public internet.
- For operational use, obtain written approval for access rules, retention/deletion, audit logs, encryption, backup, referral scripts, and data-protection procedures.
