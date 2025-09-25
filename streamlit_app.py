# streamlit_app_full_with_replies_and_reminders.py
import io
import re
import time
import uuid
import ssl
import imaplib
import email
from email import policy
from email.header import Header
from email.utils import formataddr, parseaddr, make_msgid
from collections import defaultdict
from datetime import datetime

import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- Page config ----------------
st.set_page_config(page_title="Email Automation Tool with Reminders & Auto-Reply Detection")

# --- SMTP Settings (Gmail by default) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
USE_TLS = True

# Default IMAP (Gmail)
DEFAULT_IMAP_SERVER = "imap.gmail.com"
DEFAULT_IMAP_PORT = 993

# ---------------- Helpers ----------------
def clean_value(val):
    """Clean individual cell values (remove invisible characters)."""
    if isinstance(val, str):
        return (
            val.replace("\xa0", " ")
               .replace("\u200b", "")
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

def strip_non_ascii(s: str) -> str:
    """Remove non-ASCII characters safely."""
    if not isinstance(s, str):
        return s
    return ''.join(ch if ord(ch) < 128 else ' ' for ch in s)

def imap_date_from_timestamp(ts):
    """Return an IMAP-compatible SINCE date string from UNIX timestamp."""
    dt = datetime.utcfromtimestamp(ts)
    return dt.strftime("%d-%b-%Y")  # e.g., 01-Jan-2023

def parse_email_address(header_val):
    """Return email address from header like 'Name <addr@domain>'"""
    try:
        _, addr = parseaddr(header_val)
        return addr.lower()
    except Exception:
        return header_val.lower() if isinstance(header_val, str) else ""

# ---------------- Session-state defaults ----------------
if "sent_count" not in st.session_state:
    st.session_state.sent_count = 0
if "sent_log" not in st.session_state:
    # Each entry: uid, email, name, subject, sent_at, msg_id, replied (bool), bounced (bool)
    st.session_state.sent_log = []
if "failed_rows" not in st.session_state:
    st.session_state.failed_rows = []
if "wait_time" not in st.session_state:
    st.session_state.wait_time = 20

# --- Top controls: live counter + wait time (placed above upload) ---
st.title("Email Automation Tool with Reminders & Auto-Reply Detection")

col_top1, col_top2 = st.columns([1, 1])
with col_top1:
    sent_count_placeholder = st.empty()
    sent_count_placeholder.metric("Emails sent", st.session_state.sent_count)
with col_top2:
    wait_time = st.number_input(
        "Wait time between sends (seconds)",
        min_value=1,
        max_value=600,
        value=st.session_state.wait_time,
        step=1,
        key="wait_time_input"
    )
st.session_state.wait_time = wait_time

# ---------------- Upload & Sample CSV ----------------
st.subheader("Upload recipient list")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")

# Sample CSV for user convenience
sample_df = pd.DataFrame({
    "email": ["john.doe@example.com", "jane.smith@example.com"],
    "name": ["John Doe", "Jane Smith"],
    "company": ["Acme Corp", "Globex Inc"]
})
buf = io.StringIO()
sample_df.to_csv(buf, index=False)
st.download_button("📥 Download sample CSV", data=buf.getvalue(),
                   file_name="sample_recipients.csv", mime="text/csv",
                   key="download_sample_csv")

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
        st.success("CSV uploaded and cleaned successfully ✅")
        st.dataframe(df)

# ---------------- Email Config ----------------
st.subheader("Email configuration")
from_email = st.text_input("Your email address", key="from_email")
app_password = st.text_input("App password (or SMTP app password)", type="password", key="app_password")
from_name = st.text_input("Your name (optional)", key="from_name")
# allow override of IMAP server for replies
st.markdown("IMAP (for reply detection)")
imap_server = st.text_input("IMAP server", value=DEFAULT_IMAP_SERVER, key="imap_server")
imap_port = st.number_input("IMAP port", min_value=1, max_value=65535, value=DEFAULT_IMAP_PORT, key="imap_port")

st.subheader("Cost Associated")
currency = st.selectbox("Currency", ["USD", "AED"], key="currency_select")
cost = st.number_input(f"Cost in {currency}", min_value=0.0, step=50.0, value=1000.0, key="cost_input")

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
    height=300,
    help="Use placeholders like {name}, {company}, {sender}, {cost}, {currency}",
    key="body_input"
)

# ---------------- Send & Reset Buttons ----------------
col1, col2 = st.columns(2)
with col1:
    send_clicked = st.button("🚀 Send Emails", key="send_emails_btn")
with col2:
    reset_clicked = st.button("🔄 Reset", key="reset_btn")

# Handle reset
if reset_clicked:
    for key in ["sent_count", "sent_log", "failed_rows"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# ---------------- Sending flow ----------------
if send_clicked:
    # basic validation
    if not from_email or not app_password:
        st.error("Please provide your email and app password.")
        st.stop()
    if df is None:
        st.error("Please upload a CSV file with recipients.")
        st.stop()

    # prepare
    progress = st.progress(0)
    total = len(df)
    sent = 0
    skipped_rows = []
    failed_rows = []

    # Ensure sent_log exists
    if "sent_log" not in st.session_state:
        st.session_state.sent_log = []

    # iterate
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
        rowd.setdefault("cost", str(cost))
        rowd.setdefault("currency", currency)
        rowd.setdefault("company", "")
        rowd.setdefault("name", "")

        # Construct subject and body
        subj_text = safe_format(subject_tpl, rowd)
        body_text = safe_format(body_tpl, rowd)

        # Build message
        msg = MIMEMultipart()
        from_display = clean_value(from_name or "")
        to_display = clean_value(rowd.get("name", "") or "")

        from_header = formataddr((str(Header(from_display, "utf-8")), from_email))
        to_header = formataddr((str(Header(to_display, "utf-8")), recip_addr))

        msg["From"] = from_header
        msg["To"] = to_header
        msg["Subject"] = str(Header(subj_text, "utf-8"))
        # Ensure replies land in monitored mailbox
        msg["Reply-To"] = from_email

        # Request read receipts (best-effort; often ignored)
        msg["Disposition-Notification-To"] = from_email
        msg["Return-Receipt-To"] = from_email
        msg["X-Confirm-Reading-To"] = from_email

        # Create and add message-id so we can correlate replies
        msg_id = make_msgid(domain=None)  # library-generated unique id
        msg["Message-ID"] = msg_id

        # Attach plain text (or swap to HTML if desired)
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        # Send
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

            # --- Success actions ---
            sent += 1
            st.session_state.sent_count += 1
            try:
                sent_count_placeholder.metric("Emails sent", st.session_state.sent_count)
            except Exception:
                st.write(f"Emails sent: {st.session_state.sent_count}")

            st.success(f"✅ Sent to {recip_addr}")

            # log the sent message (for reminders & reply detection)
            st.session_state.sent_log.append({
                "uid": uuid.uuid4().hex,
                "email": recip_addr.lower(),
                "name": rowd.get("name", ""),
                "subject": subj_text,
                "sent_at": int(time.time()),
                "msg_id": msg_id,   # include angle brackets since make_msgid returns that
                "replied": False,
                "bounced": False
            })

        except Exception as e:
            # record failure so it won't be included in sent_log
            st.error(f"❌ Failed to send to {recip_addr}: {e}")
            failed_rows.append({**rowd, "__reason": str(e)})
            st.session_state.failed_rows.append({**rowd, "__reason": str(e)})

        progress.progress((idx + 1) / total)

        # wait between sends
        if idx < total - 1:
            wt = int(st.session_state.get("wait_time", 20))
            countdown_placeholder = st.empty()
            for remaining in range(wt, 0, -1):
                countdown_placeholder.info(f"⏳ Waiting {remaining} seconds before next email...")
                time.sleep(1)
            countdown_placeholder.empty()

    st.info(f"✅ Done — attempted {total}, sent {sent}, skipped {len(skipped_rows)}, failed {len(failed_rows)}")

    # Offer downloads: skipped, failed, sent_log
    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        buf_skipped = io.StringIO()
        skipped_df.to_csv(buf_skipped, index=False)
        st.download_button("📥 Download skipped rows", data=buf_skipped.getvalue(),
                           file_name="skipped_recipients.csv", mime="text/csv",
                           key="download_skipped")
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        buf_failed = io.StringIO()
        failed_df.to_csv(buf_failed, index=False)
        st.download_button("📥 Download failed rows", data=buf_failed.getvalue(),
                           file_name="failed_recipients.csv", mime="text/csv",
                           key="download_failed")
    if st.session_state.get("sent_log"):
        sent_df = pd.DataFrame(st.session_state.sent_log)
        buf_sent = io.StringIO()
        sent_df.to_csv(buf_sent, index=False)
        st.download_button("📥 Download sent log (for reminders / record-keeping)", data=buf_sent.getvalue(),
                           file_name="sent_log.csv", mime="text/csv",
                           key="download_sent_log")

# ---------------- Reminders & Auto-Reply detection UI ----------------
st.subheader("Reminders & Auto-Reply Detection")

# Option to upload an external sent_log (if user wants)
use_current_log = st.checkbox("Use sent log from this session", value=True)
uploaded_sent_log = None
sent_log_df = None
if not use_current_log:
    uploaded_sent_log = st.file_uploader("Upload sent_log.csv (if not using session log)", type=["csv"], key="upload_sent_log")
    if uploaded_sent_log:
        try:
            sent_log_df = pd.read_csv(uploaded_sent_log)
            # Normalize columns if needed
            if "email" in sent_log_df.columns:
                sent_log_df["email"] = sent_log_df["email"].astype(str).str.lower()
        except Exception as e:
            st.error(f"Couldn't read uploaded sent_log: {e}")

if use_current_log:
    if st.session_state.get("sent_log"):
        sent_log_df = pd.DataFrame(st.session_state.sent_log)
    else:
        sent_log_df = pd.DataFrame(columns=["uid", "email", "name", "subject", "sent_at", "msg_id", "replied", "bounced"])

# Optionally upload a replies CSV to mark replied items
replies_file = st.file_uploader("(Optional) Upload replies CSV to mark 'replied' (columns: email or uid)", type=["csv"], key="upload_replies")

# Reminder templates
rem_subj_tpl = st.text_input("Reminder subject template", value="Following up on: {subject}", key="rem_subj")
rem_body_tpl = st.text_area("Reminder body template", value="Hi {name},\n\nJust checking in on my previous email about \"{subject}\". Would you be available for a quick chat?\n\nBest,\n{sender}", height=200, key="rem_body")

only_non_replied = st.checkbox("Only send reminders to recipients not marked as replied", value=True)
exclude_bounced = st.checkbox("Exclude recipients known to have bounced", value=True)

# Button to run IMAP-based auto-reply detection and bounce detection
with st.expander("Auto-Reply & Bounce Detection (IMAP)"):
    st.write("This will connect to your IMAP mailbox and attempt to identify replies and bounces related to messages in the sent log.")
    run_imap = st.button("Run Auto-Reply & Bounce Detection (IMAP)", key="run_imap")

def run_imap_detection(imap_server, imap_port, username, password, sent_log_entries):
    """
    Connects to IMAP and attempts to mark entries in sent_log_entries as replied or bounced.
    Returns updated sent_log_entries (list of dicts) and a list of detected bounces.
    """
    updated_log = sent_log_entries.copy()
    detected_bounces = set()
    try:
        st.info("Connecting to IMAP...")
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(imap_server, int(imap_port), ssl_context=context) as M:
            M.login(username, password)
            M.select("INBOX")
            if not updated_log:
                st.warning("No sent_log entries to check.")
                return updated_log, list(detected_bounces)

            # Map emails to their sent entries for quick lookup
            email_map = {}
            earliest_ts = min([e["sent_at"] for e in updated_log if e.get("sent_at")], default=int(time.time()) - 7*24*3600)
            for e in updated_log:
                email_map.setdefault(e["email"].lower(), []).append(e)

            # Fetch all messages since earliest sent date to reduce search space
            since_date = imap_date_from_timestamp(max(earliest_ts - 86400, 0))
            st.info(f"Searching mailbox since {since_date} ...")
            typ, data = M.search(None, f'(SINCE "{since_date}")')
            if typ != "OK":
                st.warning("IMAP search failed or returned no messages.")
                return updated_log, list(detected_bounces)

            msg_nums = data[0].split()
            st.info(f"Scanning {len(msg_nums)} messages from mailbox for replies / bounces (may take a while).")

            # Precompile patterns for bounce detection
            bounce_subject_re = re.compile(r"(?i)(undeliver|delivery failure|returned to sender|failure notice|delivery status notification|undelivered|bounce)")
            mailer_daemon_re = re.compile(r"mailer-daemon|postmaster|no-reply", re.I)

            for i, num in enumerate(msg_nums):
                if (i % 100) == 0:
                    st.progress(i / max(len(msg_nums), 1))
                typ, msg_data = M.fetch(num, '(RFC822.HEADER RFC822.TEXT)')
                if typ != "OK":
                    continue
                raw = msg_data[0][1]
                try:
                    parsed = email.message_from_bytes(raw, policy=policy.default)
                except Exception:
                    continue

                # Headers
                from_hdr = parsed.get("From", "")
                subject_hdr = parsed.get("Subject", "")
                in_reply_to = parsed.get("In-Reply-To", "")
                references = parsed.get("References", "")

                from_addr = parse_email_address(from_hdr)

                # Check bounce heuristics: from mailer-daemon/postmaster or bounce-like subject
                if mailer_daemon_re.search(from_hdr) or bounce_subject_re.search(str(subject_hdr)):
                    # try to find email addresses in body that match sent recipients
                    # fetch full body if not already
                    body = ""
                    try:
                        if parsed.is_multipart():
                            for part in parsed.walk():
                                if part.get_content_type() == "text/plain":
                                    body += part.get_content()
                        else:
                            body = parsed.get_content()
                    except Exception:
                        body = ""

                    found_emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", body)
                    found_emails = [e.lower() for e in found_emails]
                    for fe in found_emails:
                        # mark as bounced if exists in log
                        for entry in updated_log:
                            if entry["email"].lower() == fe:
                                entry["bounced"] = True
                                detected_bounces.add(fe)
                    # continue scanning
                    continue

                # Check if this message appears to be a reply to any entry
                # Strategy: match In-Reply-To to msg_id OR match from address + subject contains original subject
                candidate_msg_id = in_reply_to.strip() if in_reply_to else ""
                candidate_refs = references.strip() if references else ""
                # Also fetch body and plain text subject normalized
                subj_text = str(subject_hdr or "").lower()

                # full body text
                body = ""
                try:
                    if parsed.is_multipart():
                        for part in parsed.walk():
                            ct = part.get_content_type()
                            if ct == "text/plain":
                                body += part.get_content()
                    else:
                        body = parsed.get_content()
                except Exception:
                    body = ""

                # Attempt to match to any sent entry
                for entry in updated_log:
                    entry_email = entry["email"].lower()
                    entry_msgid = (entry.get("msg_id") or "").strip()
                    entry_subject = (entry.get("subject") or "").lower()

                    # Match by In-Reply-To / References header:
                    if entry_msgid and entry_msgid in (candidate_msg_id or ""):
                        entry["replied"] = True
                    elif entry_msgid and entry_msgid in (candidate_refs or ""):
                        entry["replied"] = True
                    # Match by from address and subject similarity
                    elif from_addr and from_addr == entry_email:
                        # subject match heuristic: whether the sent subject words are in reply subject
                        # Lowercase and check overlap
                        if entry_subject and any(word for word in entry_subject.split() if word and word in subj_text):
                            entry["replied"] = True
                        else:
                            # also check body contains some unique tokens (email address or msgid)
                            if entry_msgid and entry_msgid.strip("<>") in body:
                                entry["replied"] = True

            st.progress(1.0)
            M.logout()
    except imaplib.IMAP4.error as imap_err:
        st.error(f"IMAP error: {imap_err}")
    except Exception as exc:
        st.error(f"Error during IMAP detection: {exc}")

    return updated_log, list(detected_bounces)

if run_imap:
    if not from_email or not app_password:
        st.error("Please provide your email and app password for IMAP.")
    else:
        # Prepare local sent_log list
        current_sent_log = sent_log_df.to_dict("records") if isinstance(sent_log_df, pd.DataFrame) else []
        updated_log, detected_bounces = run_imap_detection(imap_server, imap_port, from_email, app_password, current_sent_log)

        # Update session state with updated replies/bounces
        # If using session log, update that directly; otherwise prepare updated_df for download
        if use_current_log:
            # Map by uid or email to update session log
            for upd in updated_log:
                for i, e in enumerate(st.session_state.sent_log):
                    if e.get("uid") == upd.get("uid"):
                        st.session_state.sent_log[i].update(upd)
            st.success("Auto-reply/bounce detection complete — session sent_log updated.")
        else:
            # Provide updated CSV for download
            updated_df = pd.DataFrame(updated_log)
            buf_upd = io.StringIO()
            updated_df.to_csv(buf_upd, index=False)
            st.download_button("📥 Download updated sent_log with replies/bounces", data=buf_upd.getvalue(),
                               file_name="updated_sent_log.csv", mime="text/csv", key="download_updated_log")
            st.success("Auto-reply/bounce detection complete — download the updated sent_log.")

        if detected_bounces:
            st.warning(f"Detected bounce addresses: {', '.join(detected_bounces)}")

# If user uploaded replies file, mark entries accordingly
if replies_file is not None and sent_log_df is not None:
    try:
        replies_df = pd.read_csv(replies_file)
        if "uid" in replies_df.columns:
            replied_uids = set(replies_df["uid"].astype(str).tolist())
            for entry in st.session_state.sent_log:
                if str(entry.get("uid")) in replied_uids:
                    entry["replied"] = True
        elif "email" in replies_df.columns:
            replied_emails = set(replies_df["email"].astype(str).str.lower().tolist())
            for entry in st.session_state.sent_log:
                if entry.get("email", "").lower() in replied_emails:
                    entry["replied"] = True
        else:
            st.warning("Replies file must include a column named 'uid' or 'email' to match.")
        st.success("Replied items marked in session sent_log.")
    except Exception as e:
        st.error(f"Could not process replies file: {e}")

# Show preview of recipients for reminder
if isinstance(sent_log_df, pd.DataFrame) and not sent_log_df.empty:
    st.write("Preview of sent-log (first 10):")
    st.dataframe(sent_log_df.head(10))

# ---------------- Reminders sending flow ----------------
if st.button("Send Reminders"):
    if rem_body_tpl.strip() == "" or rem_subj_tpl.strip() == "":
        st.error("Please provide reminder subject and body templates.")
        st.stop()

    # Rebuild the DataFrame to remind from session or uploaded
    if use_current_log:
        remind_df = pd.DataFrame(st.session_state.sent_log)
    else:
        remind_df = sent_log_df.copy() if isinstance(sent_log_df, pd.DataFrame) else pd.DataFrame()

    if remind_df is None or remind_df.empty:
        st.error("No sent_log available to send reminders.")
        st.stop()

    # Ensure lower-case emails
    if "email" in remind_df.columns:
        remind_df["email"] = remind_df["email"].astype(str).str.lower()

    # Exclude those flagged replied if requested
    if only_non_replied and "replied" in remind_df.columns:
        remind_df = remind_df[~remind_df["replied"].astype(bool)]

    # Exclude bounced if requested or based on session failed rows
    if exclude_bounced:
        # Remove entries marked bounced in sent_log
        if "bounced" in remind_df.columns:
            remind_df = remind_df[~remind_df["bounced"].astype(bool)]
        # Also exclude emails that failed to send earlier in session
        failed_emails = {clean_email_address(r.get("email","")).lower() for r in st.session_state.failed_rows if r.get("email")}
        remind_df = remind_df[~remind_df["email"].isin(failed_emails)]

    total_rem = len(remind_df)
    if total_rem == 0:
        st.info("No recipients to remind after filters.")
        st.stop()

    progress_rem = st.progress(0)
    sent_rem = 0
    failed_rem = []

    for idx, r in remind_df.iterrows():
        rdict = r.to_dict()
        recip = rdict.get("email")
        if not recip or "@" not in recip:
            failed_rem.append({**rdict, "__reason": "invalid email"})
            progress_rem.progress((idx + 1) / total_rem)
            continue

        # Build subject & body for reminder
        subj = safe_format(rem_subj_tpl, rdict)
        body = safe_format(rem_body_tpl, rdict)

        # Build reminder message
        msg = MIMEMultipart()
        from_header = formataddr((str(Header(clean_value(from_name or ""), "utf-8")), from_email))
        to_header = formataddr((str(Header(clean_value(rdict.get("name","") or ""), "utf-8")), recip))
        msg["From"] = from_header
        msg["To"] = to_header
        msg["Subject"] = str(Header(subj, "utf-8"))
        msg["Reply-To"] = from_email
        # optional read receipts
        msg["Disposition-Notification-To"] = from_email

        # Add a new Message-ID for the reminder
        rem_msg_id = make_msgid()
        msg["Message-ID"] = rem_msg_id

        msg.attach(MIMEText(body, "plain", "utf-8"))

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

            sent_rem += 1
            st.session_state.sent_count += 1
            try:
                sent_count_placeholder.metric("Emails sent", st.session_state.sent_count)
            except Exception:
                st.write(f"Emails sent: {st.session_state.sent_count}")
            st.success(f"✅ Reminder sent to {recip}")

            # Update session sent_log entry (mark last_reminder_at optionally)
            # find by email and mark last_reminder_sent timestamp
            for e in st.session_state.sent_log:
                if e.get("email", "").lower() == recip.lower():
                    e["last_reminder_sent_at"] = int(time.time())

        except Exception as e:
            failed_rem.append({**rdict, "__reason": str(e)})
            st.error(f"❌ Failed to send reminder to {recip}: {e}")

        progress_rem.progress((idx + 1) / total_rem)

        # wait between reminders using global wait_time
        wt = int(st.session_state.get("wait_time", 20))
        countdown_placeholder = st.empty()
        for remaining in range(wt, 0, -1):
            countdown_placeholder.info(f"⏳ Waiting {remaining} seconds before next reminder...")
            time.sleep(1)
        countdown_placeholder.empty()

    st.info(f"Reminder run finished — attempted {total_rem}, sent {sent_rem}, failed {len(failed_rem)}")
    if failed_rem:
        df_fail = pd.DataFrame(failed_rem)
        buff = io.StringIO()
        df_fail.to_csv(buff, index=False)
        st.download_button("📥 Download failed reminders", data=buff.getvalue(), file_name="failed_reminders.csv", mime="text/csv")

# ---------------- End of script UI/notes ----------------
st.markdown("""
Notes:
- Auto-reply detection uses IMAP and heuristics: In-Reply-To / References headers and From + Subject similarity.
- Bounce detection is heuristic-based (MAILER-DAEMON / bounce words in subject); it may not find every bounce.
- For more robust bounce & reply tracking, consider using a transactional email provider (SendGrid, Mailgun, Amazon SES) which provides webhooks and dashboards.
- IMAP connection requires correct server/port and credentials. For Gmail, enable app passwords and IMAP access.
""")
