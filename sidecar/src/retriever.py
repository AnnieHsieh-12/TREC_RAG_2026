import time

import requests

from config import BASE_URL, INDEX_NAME


class AuthError(Exception):
    pass


class RetrieverError(Exception):
    pass


def _get_with_connection_retry(url, params, headers, max_retries):
    """GET with exponential backoff on connection-level failures (DNS
    hiccups, dropped connections) - these are transient network blips, not
    a server-side rejection, so they should be retried rather than raised
    immediately like a 429 or 404 would be."""
    attempt = 0
    while True:
        try:
            return requests.get(url, params=params, headers=headers, timeout=30)
        except requests.exceptions.ConnectionError:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(min(10 * (2 ** (attempt - 1)), 90))


def search_climbmix(query, hits, token, parse=False, max_retries=3):
    url = f"{BASE_URL}/v1/{INDEX_NAME}/search"
    params = {"query": query, "hits": hits, "parse": "true" if parse else "false"}
    headers = {"X-API-Key": token}

    attempt = 0
    while True:
        response = _get_with_connection_retry(url, params, headers, max_retries)

        if response.status_code == 200:
            return response.json()["candidates"]

        if response.status_code == 401:
            raise AuthError(response.json().get("error", "unauthorized"))

        if response.status_code == 429:
            attempt += 1
            if attempt > max_retries:
                raise RetrieverError(f"429 rate limited after {max_retries} retries")
            wait_seconds = int(response.headers.get("Retry-After", 5))
            time.sleep(wait_seconds)
            continue

        if response.status_code in (502, 503, 504):
            attempt += 1
            if attempt > max_retries:
                raise RetrieverError(
                    f"{response.status_code} after {max_retries} retries"
                )
            time.sleep(min(10 * (2 ** (attempt - 1)), 90))
            continue

        raise RetrieverError(
            f"unexpected status {response.status_code}: {response.text[:300]}"
        )


def fetch_doc(docid, token, parse=True, max_retries=3):
    url = f"{BASE_URL}/v1/{INDEX_NAME}/doc/{docid}"
    params = {"parse": "true" if parse else "false"}
    headers = {"X-API-Key": token}

    attempt = 0
    while True:
        response = _get_with_connection_retry(url, params, headers, max_retries)

        if response.status_code == 200:
            return response.json()["doc"]

        if response.status_code == 401:
            raise AuthError(response.json().get("error", "unauthorized"))

        if response.status_code == 404:
            raise RetrieverError(f"document not found: {docid}")

        if response.status_code == 429:
            attempt += 1
            if attempt > max_retries:
                raise RetrieverError(f"429 rate limited after {max_retries} retries")
            wait_seconds = int(response.headers.get("Retry-After", 5))
            time.sleep(wait_seconds)
            continue

        if response.status_code in (502, 503, 504):
            attempt += 1
            if attempt > max_retries:
                raise RetrieverError(
                    f"{response.status_code} after {max_retries} retries"
                )
            time.sleep(min(10 * (2 ** (attempt - 1)), 90))
            continue

        raise RetrieverError(
            f"unexpected status {response.status_code}: {response.text[:300]}"
        )
