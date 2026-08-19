#!/usr/bin/env python3
"""One-off login helper for assamarchive.org ("Digitizing অসম" -- Nanda
Talukdar Foundation's Assamese book/journal digitization project).

Run this yourself -- it reads the credential from a local, gitignored file
(never hardcoded, never committed) and performs the actual authentication.
It saves the resulting session cookies to another gitignored file that the
scraper (assamarchive_download.py, once built) reads to make authenticated
requests -- the scraper itself never touches the password.

Setup: copy assamese/data/.credentials.local.json.template to
assamese/data/.credentials.local.json and fill in emailOrMobile/password.
Both files are gitignored (see repo .gitignore).

API discovered by inspecting assamarchive.org's own JS bundle (read-only,
no login attempted during discovery): base /api/v1, login endpoint
/auth/login, JSON body {emailOrMobile, password}, cookie-based session
(axios withCredentials). No CAPTCHA or 2FA on login (OTP is only used for
registration/password-reset flows).
"""

import json
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
CREDENTIALS_FILE = DATA_DIR / ".credentials.local.json"
SESSION_FILE = DATA_DIR / ".assamarchive_session.json"

BASE_URL = "https://assamarchive.org/api/v1"
LOGIN_URL = f"{BASE_URL}/auth/login"


def main():
    if not CREDENTIALS_FILE.exists():
        raise SystemExit(
            f"Missing {CREDENTIALS_FILE}. Copy the .template file next to it "
            f"and fill in emailOrMobile/password first."
        )
    creds = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))

    session = requests.Session()
    resp = session.post(
        LOGIN_URL,
        json={"emailOrMobile": creds["emailOrMobile"], "password": creds["password"]},
        headers={"User-Agent": "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"},
        timeout=30,
    )
    resp.raise_for_status()

    # Save cookies (session-based auth) plus any bearer/JWT token the login
    # response body returns, so the scraper can use whichever the API needs.
    session_data = {
        "cookies": session.cookies.get_dict(),
        "login_response_body": resp.json() if resp.content else None,
    }
    SESSION_FILE.write_text(json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Login succeeded, session saved to {SESSION_FILE}")


if __name__ == "__main__":
    main()
