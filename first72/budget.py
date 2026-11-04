"""Hard cap on live API calls, so a limited key or credit balance cannot be overspent.

Set F72_MAX_API_CALLS=5 in .env. Every outgoing model call (reasoning or SAM) is counted in a small file
(.api_calls, ignored by git) so the cap holds across restarts. When the cap is reached the call is refused
with 'API call limit reached' and the app falls back to offline mode. Unset or 0 = no cap.
Reset the counter by deleting .api_calls or by running: python -m first72 reset-budget
"""
import os

PATH = os.environ.get("F72_BUDGET_FILE", ".api_calls")


def limit():
    try:
        return int(os.environ.get("F72_MAX_API_CALLS", "0") or 0)
    except ValueError:
        return 0


def used():
    try:
        with open(PATH) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def spend():
    """Call right before every live request. Raises once the cap is reached; counts the attempt otherwise."""
    cap = limit()
    n = used()
    if cap and n >= cap:
        raise RuntimeError(f"API call limit reached ({n}/{cap}); no more live calls will be made. Delete {PATH} to reset.")
    try:
        with open(PATH, "w") as f:
            f.write(str(n + 1))
    except OSError:
        pass


def reset():
    try:
        os.remove(PATH)
    except OSError:
        pass
