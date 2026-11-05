"""Give someone a login for the staff test page. No email is sent.

    python scripts/add_tester.py harriet@sharktrust.org            # prints a password
    python scripts/add_tester.py harriet@sharktrust.org 'chosen-password'

Any email-shaped name works, e.g. harriet@eggic.test. The tester flag goes in
app_metadata, which only the secret key can set; the page's policies check it.
"""

from __future__ import annotations

import secrets
import sys

from _supabase import call


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    email = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) == 3 else secrets.token_urlsafe(9)
    call("POST", "/auth/v1/admin/users", json={
        "email": email, "password": password, "email_confirm": True,
        "app_metadata": {"tester": True},
    })
    print(f"{email} can now log in at /test")
    if len(sys.argv) == 2:
        print(f"password: {password}")


if __name__ == "__main__":
    main()
