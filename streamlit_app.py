import io
import re
import time
from collections import defaultdict
from email.header import Header
from email.utils import formataddr, parseaddr

import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title="Team Niwrutti")

# --- SMTP Settings (Gmail by default) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
USE_TLS = True

# ---------------- Helpers ----------------
def clean_value(val):
    """Clean individual cell values (remove invisible characters)."""
    if isinstance(val, str):
        return (
            val.replace("\xa0", " ")      # non-breaking space
               .replace("\u200b", "")     # zero-width space
               .strip()
        )
    return val

def clean_email_address(raw_email: str) -> str | None:
    """Parse and sanitize an email address string."""
    if not raw_email:
        return None
    raw_email = clean_value(raw_email)
    _, addr = parseaddr(raw_email)
    if not addr:
        addr = re.sub(r"[<>\s\"']", "", raw_email)
    addr = addr.strip()
    if "@" not in addr:
        return None
    try:
        local, domain = addr.rsplit("@", 1)
    except ValueError:
        return None
    try:
        domain_ascii = domain.encode("idna").decode("ascii")
    except Exception:
        domain_ascii = "".join(ch for ch in domain if ord(ch) < 128)
    return f"{local}@{domain_ascii}"

def safe_format(template: str, mapping: dict) -> str:
    """Format template safely with missing keys allowed."""
    return template.format_map(defaultdict(str, mapping))

def clean_display_name(name: str) -> str:
    """Clean and normalize display names for email headers."""
    if not name:
        return ""
    # Replace non-breaking spaces and zero-width spaces with normal space
    name = name.replace("\xa0", " ").replace("\u200b", "")
    # Strip leading/trailing spaces
    name = name.strip()
    return name

def clean_invisible_unicode(s: str) -> str:
    """Remove invisible unicode characters such as non-breaking spaces."""
    if not isinstance(s, str):
        return s
    return s.replace('\xa0', '').replace('\u200b', '').strip()

# ---------------- Upload & Sample CSV ----------------
st.title("Team Niwrutti")
st.subheader("Upload recipient list")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

# Sample CSV for download
sample_df = pd.DataFrame({
    "email": ["john.doe@example.com", "jane.smith@example.com"],
    "name": ["John Doe", "Jane Smith"],
    "company": ["Acme Corp", "Globex Inc"]
})
buf = io.StringIO()
sample_df.to_csv(buf, index=False)
st.download_button(
    "Download sample CSV",
    data=buf.getvalue(),
    file_name="sample_recipients.csv",
    mime="text/csv",
    key="download_sample_csv"
)

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            df = None
    if df is not None:
        # Clean entire DataFrame
        df = df.applymap(clean_value)
        df.columns = [clean_value(c) for c in df.columns]
        st.success("CSV cleaned and uploaded successfully")
        st.dataframe(df)

# Initialize session state variables for controlling UI
if "sending" not in st.session_state:
    st.session_state.sending = False
if "stop_sending" not in st.session_state:
    st.session_state.stop_sending = False
if "sent_count" not in st.session_state:
    st.session_state.sent_count = 0

# initialize live sent counter
if "sent_count" not in st.session_state:
    st.session_state.sent_count = 0

# ---------------- Email Config ----------------
st.subheader("Email configuration")
from_email = clean_invisible_unicode(st.text_input("Your email address", key="from_email"))
app_password = clean_invisible_unicode(st.text_input("App password", type="password", key="app_password"))
from_name = st.text_input("Your name (optional)", key="from_name")

#st.subheader("Cost Associated")
#currency = st.selectbox("Currency", ["USD", "AED"], key="currency_select")
#cost = st.number_input(f"Cost in {currency}", min_value=0.0, step=50.0, value=1000.0, key="cost_input")

# ---------------- Compose Message ----------------
st.subheader("Compose message")

subject_tpl = st.text_input(
    "Enter subject line template",
    placeholder="Paste your Subject Line (Include any placeholders if required.)",
    value="",
    help="Use placeholders like {name}, {company}, {sender}, {cost}, {currency}",
    key="subject_input"
)

body_tpl = st.text_area(
    "Enter body template",
    placeholder=("Paste Your Email Body (Include any placeholders if required.)"),
    value="",
    height=850,
    help="Use placeholders like {name}, {company}, {sender}, {cost}, {currency}",
    key="body_input"
)

# live counter placeholder (shows 0 initially)
sent_count_placeholder = st.empty()
sent_count_placeholder.metric("Emails sent", st.session_state.sent_count)

progress = st.progress(0)


# ---------------- Send & Stop Buttons ----------------
col1, col2 = st.columns(2)

with col1:
    send_clicked = st.button("Send Emails", key="send_emails_btn")

with col2:
    stop_clicked = st.button("Stop Sending", key="stop_sending_btn")

if stop_clicked:
    st.session_state.stop_sending = True

# Initialize stop flag before sending
if send_clicked:
    st.session_state.stop_sending = False
    st.session_state.sent_count = 0

    # ... your existing validation code ...

    for idx, row in df.iterrows():
        if st.session_state.get("stop_sending", False):
            st.warning("Email sending stopped by user.")
            break
    st.session_state.sending = True
    st.session_state.stop_sending = False
    st.session_state.sent_count = 0
    
    progress = st.progress(0)
    total = len(df)
    sent = 0
    skipped_rows = []
    failed_rows = []

    for idx, row in df.iterrows():
        rowd = {str(k): clean_value(v) for k, v in row.to_dict().items()}

        # Validate recipient email
        recip_addr = clean_email_address(rowd.get("email", ""))
        if not recip_addr:
            skipped_rows.append({**rowd, "__reason": "missing/invalid email"})
            progress.progress((idx + 1) / total)
            continue

        # Defaults
        rowd.setdefault("sender", from_name)
        rowd.setdefault("company", "")
        rowd.setdefault("name", "")

        # Extract first name for body only
        full_name = rowd.get("name", "")
        first_name = full_name.split()[0] if full_name.strip() else ""

        # Prepare mappings for subject and body separately
        subject_mapping = dict(rowd)  # full name for subject
        body_mapping = dict(rowd)
        body_mapping["name"] = first_name  # first name for body

        subj_text = safe_format(subject_tpl, subject_mapping)
        body_text = safe_format(body_tpl, body_mapping)

        # Build HTML body with Times New Roman font and preserve formatting
        html_body = f"""\
<html>
  <body style="font-family: 'Times New Roman', serif;">
    <pre style="font-family: 'Times New Roman', serif; white-space: pre-wrap;">{body_text}</pre>
  </body>
</html>
"""

        # Build message
        msg = MIMEMultipart()

        from_display = clean_display_name(from_name or "")
        to_display = clean_display_name(rowd.get("name", "") or "")

        from_header = formataddr((str(Header(from_display, "utf-8")), from_email))
        to_header = formataddr((str(Header(to_display, "utf-8")), recip_addr))

        msg["From"] = from_header
        msg["To"] = to_header
        msg["Subject"] = str(Header(subj_text, "utf-8"))
        # Request read receipts (client-dependent; often ignored)
        msg["Disposition-Notification-To"] = from_email
        msg["Return-Receipt-To"] = from_email
        msg["X-Confirm-Reading-To"] = from_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            if USE_TLS:
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(from_email, app_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                    server.login(from_email, app_password)
                    server.send_message(msg)
            
        # increment local and session counters
            sent += 1
            st.session_state.sent_count += 1

        # update live metric (falls back to a simple write if placeholder is missing)
            try:
                sent_count_placeholder.metric("Emails sent", st.session_state.sent_count)
            except Exception:
                st.write(f"Emails sent: {st.session_state.sent_count}")

            st.success(f"✅ Sent to {recip_addr}")
        except Exception as e:
            st.error(f"Failed to send to {recip_addr}: {e}")
            failed_rows.append({**rowd, "__reason": str(e)})

        progress.progress((idx + 1) / total)

        # --- ⏳ Wait 28s before next email ---
        wait_time = 22
        countdown_placeholder = st.empty()
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = int(wait_time - elapsed)
            if remaining <= 0:
                break
            countdown_placeholder.info(f"⏳ Waiting {remaining} seconds before next email...")
            time.sleep(1)

        countdown_placeholder.empty()

    st.info(f"Done — attempted {total}, sent {sent}, skipped {len(skipped_rows)}, failed {len(failed_rows)}")
    
    # Show total sent count separately
    #st.markdown(f"### Total Emails Sent: {sent}")
    st.markdown(f"**Total emails successfully sent:** {st.session_state.sent_count}")

    # Download skipped/failed rows if any
    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        buf_skipped = io.StringIO()
        skipped_df.to_csv(buf_skipped, index=False)
        st.download_button(
            "Download skipped rows",
            data=buf_skipped.getvalue(),
            file_name="skipped_recipients.csv",
            mime="text/csv",
            key="download_skipped"
        )
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        buf_failed = io.StringIO()
        failed_df.to_csv(buf_failed, index=False)
        st.download_button(
            "Download failed rows",
            data=buf_failed.getvalue(),
            file_name="failed_recipients.csv",
            mime="text/csv",
            key="download_failed"
        )
        
st.session_state.sending = False

