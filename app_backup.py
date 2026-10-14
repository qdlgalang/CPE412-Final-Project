
import streamlit as st
import joblib

from pathlib import Path
from datetime import datetime

# -----------------------------
# 1. Load the trained model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "service_model.joblib"

st.set_page_config(
    page_title="MARITESS",
    page_icon="💬",
    layout="wide"
)

st.title("MARITESS")
st.caption(
    "My Assistant for Requirements, Intake, "
    "Triage, Emergency, and Service Support"
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Run python train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------
# 2. Demonstration service rules
# --------------------------------
# IMPORTANT:
# Replace these sample requirements
# with the client's approved service list.
SERVICE_NAMES = {
    "barangay_clearance": "Barangay Clearance",
    "residency": "Certificate of Residency",
    "indigency": "Certificate of Indigency",
    "lupon_complaint": "Lupon Complaint",
    "vaw_report": "VAW Report",
    "child_protection": "Child Protection Report",
    "disaster_report": "Disaster Report",
}

REQUIREMENTS = {
    "barangay_clearance": [
        "Resident's full name",
        "Complete address",
        "Purpose of clearance",
        "Official documentary requirements confirmed"
    ],
    "residency": [
        "Resident's full name",
        "Complete address",
        "Purpose of certificate",
        "Official documentary requirements confirmed"
    ],
    "indigency": [
        "Resident's full name",
        "Complete address",
        "Purpose of certificate",
        "Official documentary requirements confirmed"
    ],
    "lupon_complaint": [
        "Complainant's details",
        "Nature of complaint",
        "Basic incident details",
        "Appropriate staff review"
    ],
    "vaw_report": [
        "Initial report details",
        "Authorized officer review",
        "Appropriate referral assessment"
    ],
    "child_protection": [
        "Initial report details",
        "Authorized officer review",
        "Appropriate referral assessment"
    ],
    "disaster_report": [
        "Location of incident",
        "Nature of reported incident",
        "Assistance needed"
    ]
}

# --------------------------------
# 3. Basic urgent-case rules
# --------------------------------
URGENT_PHRASES = [
    "in danger",
    "hurting me",
    "is hurting",
    "being beaten",
    "being abused",
    "domestic violence",
    "sexual assault",
    "child abuse",
    "threatened to kill",
    "wants to kill me",
    "weapon",
    "flooding",
    "trapped in flood",
    "can't breathe",
    "cannot breathe",
    "not breathing",
    "severe bleeding",
    "medical emergency",
    "fire in progress"
]

def check_urgency(text):
    text_lower = text.lower()

    matches = [
        phrase for phrase in URGENT_PHRASES
        if phrase in text_lower
    ]

    return bool(matches), matches

# --------------------------------
# 4. Staff intake form
# --------------------------------
st.subheader("Resident Intake")

st.warning(
    "Prototype only: use fictional data during testing. "
    "This version is not an emergency dispatch system."
)

with st.form("intake_form"):
    request_text = st.text_area(
        "Resident's request",
        placeholder=(
            "Example: I need a brgy clearance "
            "for employment."
        )
    )

    resident_name = st.text_input(
        "Resident's name (optional for this prototype)"
    )

    resident_address = st.text_input(
        "Resident's address (optional for this prototype)"
    )

    purpose = st.text_input(
        "Purpose of request"
    )

    submitted = st.form_submit_button(
        "Analyze Request"
    )

# --------------------------------
# 5. Analyze request
# --------------------------------
if submitted:
    if not request_text.strip():
        st.error("Please enter the resident's request.")
        st.stop()

    # Urgency check comes before normal processing.
    is_urgent, matched_phrases = check_urgency(
        request_text
    )

    if is_urgent:
        st.error("URGENT REVIEW REQUIRED")

        st.write(
            "The request contains wording that matches "
            "an urgent-case rule."
        )

        st.write("Matched indicators:", ", ".join(
            matched_phrases
        ))

        st.info(
            "Staff must assess the situation immediately "
            "and follow approved barangay referral "
            "procedures. For immediate danger, contact "
            "the appropriate emergency service. "
            "Quezon City Helpline 122 is a configured "
            "referral option to verify with the client."
        )

    # Predict the requested service.
    probabilities = model.predict_proba([request_text])[0]
    classes = model.classes_

    best_index = probabilities.argmax()
    predicted_label = classes[best_index]
    confidence = float(probabilities[best_index])

    st.subheader("Analysis Results")

    if confidence < 0.45:
        st.warning(
            "The service prediction is uncertain. "
            "Please clarify the resident's request "
            "before selecting a service."
        )
    else:
        st.success(
            "Predicted service: "
            + SERVICE_NAMES.get(
                predicted_label, predicted_label
            )
        )

    st.caption(
        f"Model score: {confidence:.1%}. "
        "This is not a calibrated probability of correctness."
    )

    # Show sample requirement checklist.
    st.subheader("Requirement Checklist")

    requirements = REQUIREMENTS.get(
        predicted_label, []
    )

    if requirements:
        checked_items = []

        for item in requirements:
            checked = st.checkbox(
                item,
                key=f"{predicted_label}_{item}"
            )

            if checked:
                checked_items.append(item)

        missing_items = [
            item for item in requirements
            if item not in checked_items
        ]

        if missing_items:
            st.warning("Items still requiring confirmation:")
            for item in missing_items:
                st.write(f"- {item}")
        else:
            st.success(
                "All checklist items have been marked "
                "as confirmed by staff."
            )

    # Draft summary.
    st.subheader("Draft Case Summary")

    summary = f"""
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    Resident: {resident_name or "Not provided"}
    Address: {resident_address or "Not provided"}
    Service: {SERVICE_NAMES.get(predicted_label, predicted_label)}
    Purpose: {purpose or "Not provided"}
    Request: {request_text}
    Urgent indicator: {"Yes - staff review required" if is_urgent else "No keyword rule matched"}
    """

    st.text_area(
        "Review and edit the summary",
        value=summary.strip(),
        height=220,
        key="case_summary"
    )

    st.info(
        "This prototype does not yet save cases to a "
        "database. Checklist entries and model predictions "
        "must be reviewed by authorized staff."
    )

