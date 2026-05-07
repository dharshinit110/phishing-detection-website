"""
Feature extraction utilities for phishing URL detection.

This module converts raw URL strings into a fixed set of numeric
features that can be consumed by scikit-learn models.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import List

import numpy as np
import pandas as pd

SHORTENING_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
    "rebrand.ly",
}


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "update",
    "bank",
]


def has_ip_address(netloc: str) -> int:
    """
    Check whether the hostname part of the URL is an IPv4 address.
    Very simple heuristic using regex.
    """
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return int(bool(re.match(ip_pattern, netloc)))


def count_subdomains(netloc: str) -> int:
    """
    Count the number of subdomains in the hostname.
    Example:
        'login.paypal.com' -> 1
        'a.b.example.co.uk' -> 3
    """
    # Remove port if present
    netloc = netloc.split(":")[0]
    parts = netloc.split(".")
    # Heuristic: domain + TLD are at least 2 parts
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def contains_suspicious_keyword(url: str) -> int:
    """
    Check whether the URL string contains any of the suspicious words.
    """
    url_lower = url.lower()
    return int(any(keyword in url_lower for keyword in SUSPICIOUS_KEYWORDS))


def extract_url_features(url: str) -> List[float]:
    """
    Extract a list of numeric features from a single URL string.

    Features:
        0. URL length
        1. Number of dots ('.')
        2. Presence of '@' symbol (0/1)
        3. Presence of IP address in domain (0/1)
        4. HTTPS usage (scheme == 'https') (0/1)
        5. Number of subdomains
        6. Presence of '-' in domain (0/1)
        7. Presence of suspicious keywords in URL (0/1)
    """
    if not isinstance(url, str) or not url.strip():
        # Return a vector of zeros for clearly invalid URLs
        return [0.0] * 8

    parsed = urlparse(url.strip())

    url_length = len(url)
    num_dots = url.count(".")
    has_at_symbol = int("@" in url)

    netloc = parsed.netloc if parsed.netloc else parsed.path
    ip_in_domain = has_ip_address(netloc)

    uses_https = int(parsed.scheme.lower() == "https")
    subdomain_count = count_subdomains(netloc) if netloc else 0
    has_hyphen_in_domain = int("-" in netloc) if netloc else 0
    suspicious_kw = contains_suspicious_keyword(url)

    return [
        float(url_length),
        float(num_dots),
        float(has_at_symbol),
        float(ip_in_domain),
        float(uses_https),
        float(subdomain_count),
        float(has_hyphen_in_domain),
        float(suspicious_kw),
    ]


def _uci_url_length_category(url_length: int) -> int:
    """
    UCI-style encoding for URL_Length:
      1  => legitimate (short)
      0  => suspicious (medium)
     -1  => phishing (long)

    Common thresholds used with the UCI phishing dataset:
      < 54 => 1
      54-75 => 0
      > 75 => -1
    """
    if url_length < 54:
        return 1
    if url_length <= 75:
        return 0
    return -1


def extract_uci_url_subset_features(url: str) -> List[int]:
    """
    Extract a subset of features compatible with the UCI 'Phishing Websites' dataset,
    using only information derivable from the URL string.

    Output order (8 features):
      0 having_IP_Address       {-1, 1}
      1 URL_Length              {1, 0, -1}
      2 Shortining_Service      {1, -1}
      3 having_At_Symbol        {1, -1}
      4 Prefix_Suffix           {-1, 1}
      5 having_Sub_Domain       {-1, 0, 1}
      6 SSLfinal_State          {-1, 0, 1}  (approximated from scheme)
      7 HTTPS_token             {-1, 1}

    Notes:
    - The original UCI dataset includes many non-URL features (traffic, rank, etc.).
      This function intentionally limits to URL-derivable signals so the Streamlit app
      can still accept a raw URL.
    """
    if not isinstance(url, str) or not url.strip():
        return [1, 1, 1, 1, 1, 1, -1, 1]

    url = url.strip()
    parsed = urlparse(url)
    netloc = parsed.netloc if parsed.netloc else parsed.path
    hostname = (netloc.split(":")[0] if netloc else "").lower()

    # 0 having_IP_Address
    having_ip_address = -1 if has_ip_address(hostname) else 1

    # 1 URL_Length category
    url_length_cat = _uci_url_length_category(len(url))

    # 2 Shortining_Service
    short_service = -1 if hostname in SHORTENING_DOMAINS else 1

    # 3 having_At_Symbol
    at_symbol = -1 if "@" in url else 1

    # 4 Prefix_Suffix (hyphen in domain)
    prefix_suffix = -1 if "-" in hostname else 1

    # 5 having_Sub_Domain category
    subdomains = count_subdomains(hostname) if hostname else 0
    if subdomains == 0:
        sub_domain_cat = 1
    elif subdomains == 1:
        sub_domain_cat = 0
    else:
        sub_domain_cat = -1

    # 6 SSLfinal_State (approx)
    scheme = (parsed.scheme or "").lower()
    if scheme == "https":
        ssl_final_state = 1
    elif scheme == "http":
        ssl_final_state = -1
    else:
        ssl_final_state = 0

    # 7 HTTPS_token: 'https' in the domain part is suspicious in UCI definition
    https_token = -1 if "https" in hostname else 1

    return [
        having_ip_address,
        url_length_cat,
        short_service,
        at_symbol,
        prefix_suffix,
        sub_domain_cat,
        ssl_final_state,
        https_token,
    ]


def build_feature_matrix(urls: List[str]) -> np.ndarray:
    """
    Convert a list of URLs into a 2D numpy array of features.
    """
    features = [extract_url_features(u) for u in urls]
    return np.asarray(features, dtype=float)


def add_features_to_dataframe(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    """
    Given a DataFrame with a URL column, append feature columns and return it.
    This is convenient for inspection and debugging.
    """
    if url_column not in df.columns:
        raise ValueError(f"URL column '{url_column}' not found in DataFrame.")

    feature_names = [
        "url_length",
        "num_dots",
        "has_at_symbol",
        "ip_in_domain",
        "uses_https",
        "subdomain_count",
        "hyphen_in_domain",
        "suspicious_keyword",
    ]

    feature_matrix = build_feature_matrix(df[url_column].tolist())

    for idx, name in enumerate(feature_names):
        df[name] = feature_matrix[:, idx]

    return df


__all__ = [
    "extract_url_features",
    "extract_uci_url_subset_features",
    "build_feature_matrix",
    "add_features_to_dataframe",
]

