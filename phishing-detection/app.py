"""
Streamlit web application for phishing website detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import textwrap

import streamlit as st

from src.predict import predict_url, ModelNotFoundError
from src.domain_reputation import check_domain_reputation


def set_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Phishing Website Detection",
        page_icon="🔒",
        layout="centered",
    )


def render_header() -> None:
    """Render the main page header and description."""
    st.title("Phishing Website Detection")
    st.write(
        textwrap.dedent(
            """
            This tool uses a machine learning model to predict whether a website URL
            is **legitimate** or **phishing** based on URL characteristics.
            """
        )
    )


def render_input_section():
    """Render the URL input and prediction section."""
    st.subheader("Check a Website")
    url = st.text_input(
        "Enter a website URL",
        placeholder="e.g. https://www.your-bank.com/login",
    )

    check_button = st.button("Check Website")

    if check_button:
        if not url.strip():
            st.warning("Please enter a valid URL.")
            return

        try:
            label, confidence = predict_url(url)
            # Extra module: domain reputation & existence + HTTP reachability
            rep = check_domain_reputation(url)
        except ModelNotFoundError as exc:
            st.error(
                "Trained model not found. Please train the model first by running:\n\n"
                "`python -m src.train_model`"
            )
            st.caption(str(exc))
            return
        except Exception as exc:  # Fallback error handling
            st.error("An unexpected error occurred while making the prediction.")
            st.caption(str(exc))
            return

        # If both DNS and HTTP checks fail and the model says legitimate,
        # treat the site as suspicious/phishing (likely non-existent or typo).
        domain_override = False
        if (not rep.resolves and not rep.http_reachable) and label == 0:
            label = 1
            domain_override = True

        if label == 1:
            if domain_override:
                st.error(
                    "🚨 Suspicious Website detected!\n\n"
                    "The domain appears unreachable (DNS and HTTP failed). "
                    "This often indicates a non-existent or mistyped website, "
                    "so it is treated as phishing."
                )
            else:
                st.error("🚨 Phishing Website detected!")
        else:
            st.success("✅ Legitimate Website.")

        if confidence > 0.0:
            st.caption(f"Model confidence: {confidence:.2%}")

        # Domain reputation & existence module
        with st.expander("Domain Reputation & Existence Check", expanded=True):
            if rep.resolves:
                st.write(f"**Domain**: `{rep.domain}` (DNS resolved)")
            else:
                st.write(f"**Domain**: `{rep.domain}` (DNS did not resolve)")
                if rep.resolves_error:
                    st.caption(f"Resolution error: {rep.resolves_error}")

            if rep.http_reachable:
                st.write("**HTTP status**: reachable (2xx/3xx response)")
            else:
                st.write("**HTTP status**: not reachable")
                if rep.http_error:
                    st.caption(f"HTTP error: {rep.http_error}")

            # Highlight that partial network failures might be local issues.
            if (not rep.resolves) or (not rep.http_reachable):
                st.warning(
                    "Network checks failed (DNS and/or HTTP). This may be due to "
                    "local connectivity, firewall, or proxy issues. When *both* "
                    "DNS and HTTP fail, the app treats the site as suspicious."
                )

            st.write(
                f"**Length**: {rep.length} characters, "
                f"**name entropy**: {rep.label_entropy:.2f} bits"
            )

            flags = []
            if rep.is_very_long:
                flags.append("very long domain")
            if rep.is_high_entropy:
                flags.append("high‑entropy / random‑looking label")

            if flags:
                st.warning(
                    "Heuristic flags: " + ", ".join(flags) +
                    ". This does not guarantee phishing, but it is suspicious."
                )
            else:
                st.info(
                    "No strong lexical red flags detected for the domain "
                    "(this does **not** guarantee safety)."
                )


def render_footer() -> None:
    """Render a small footer with guidance."""
    st.markdown("---")
    st.caption(
        "This demo is for educational purposes only. "
        "Always exercise caution when visiting unknown websites."
    )


def main() -> None:
    set_page_config()
    render_header()
    render_input_section()
    render_footer()


if __name__ == "__main__":
    main()

