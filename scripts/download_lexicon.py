"""Download the Warriner et al. (2013) norms as a NovaVision lexicon.

The norms ship as the paper's Springer supplementary zip (the old
``crr.ugent.be`` CSV path is dead). The download is fetched over HTTPS and its
SHA-256 is checked against a pinned digest, so a tampered or silently re-issued
file fails loudly instead of corrupting every affect score derived from it.
Override ``--url``/``--sha256`` for a mirror, or ``--no-verify`` to skip the
check (not recommended for research use).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Warriner et al. (2013), "Norms of valence, arousal, and dominance for 13,915
# English lemmas", Behavior Research Methods, supplementary material (ESM 1).
WARRINER_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
)
WARRINER_SHA256 = "5a0db0437ef234219fb899caced5eb3195dbf9f84396637ab4db74c50ca441aa"
CSV_MEMBER = "BRM-emot-submit.csv"


class _HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Reject any redirect that would downgrade the transport off HTTPS.

    urllib follows http/ftp redirect targets by default, so validating only the
    initial URL leaves a MITM free to bounce the fetch to plaintext. Every hop
    must stay HTTPS.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise ValueError(f"refusing a non-HTTPS redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_https(url: str) -> bytes:
    """GET ``url`` over HTTPS, refusing any downgrade on the initial or redirected hop."""
    if not url.lower().startswith("https://"):
        raise ValueError("refusing a non-HTTPS URL; the norms back research results")
    opener = urllib.request.build_opener(_HTTPSOnlyRedirect)
    with opener.open(url) as resp:  # noqa: S310 - HTTPS enforced on every hop above
        return resp.read()


def verify_sha256(blob: bytes, expected: str) -> str:
    """Return the digest; raise if it does not match a non-empty ``expected``."""
    got = hashlib.sha256(blob).hexdigest()
    if expected and got != expected:
        raise ValueError(
            f"checksum mismatch: expected {expected}, got {got}. "
            "Refusing a tampered or silently updated file; pass --sha256 to accept it."
        )
    return got


def extract_csv(zip_bytes: bytes, member: str = CSV_MEMBER) -> str:
    """Read one CSV member out of the norms zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.read(member).decode("utf-8", errors="replace")


def convert(raw: str) -> list[tuple[str, float, float]]:
    """Warriner 1..9 valence/arousal to NovaVision valence (-1..1)/arousal (0..1)."""
    rows = []
    for row in csv.DictReader(io.StringIO(raw)):
        word = row["Word"].strip().lower()
        valence = (float(row["V.Mean.Sum"]) - 5.0) / 4.0
        arousal = (float(row["A.Mean.Sum"]) - 1.0) / 8.0
        rows.append((word, round(valence, 4), round(arousal, 4)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Warriner affect norms")
    parser.add_argument("--out", default="data/lexicon/warriner.tsv")
    parser.add_argument("--url", default=WARRINER_URL, help="norms zip URL (HTTPS)")
    parser.add_argument("--member", default=CSV_MEMBER, help="CSV filename inside the zip")
    parser.add_argument(
        "--sha256",
        default=WARRINER_SHA256,
        help="expected SHA-256 of the downloaded zip (pinned; override for a mirror)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the integrity check (not recommended)",
    )
    args = parser.parse_args()

    blob = fetch_https(args.url)
    if args.no_verify:
        print("WARNING: --no-verify set; skipping integrity check.", file=sys.stderr)
    else:
        if not args.sha256:
            parser.error("empty --sha256; pass a digest or use --no-verify to skip explicitly")
        verify_sha256(blob, args.sha256)
    rows = convert(extract_csv(blob, args.member))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("word\tvalence\tarousal\n")
        for word, valence, arousal in rows:
            fh.write(f"{word}\t{valence}\t{arousal}\n")

    print(f"Wrote {out} ({len(rows)} words). Set NOVAVISION_LEXICON={out} to use it.")


if __name__ == "__main__":
    main()
