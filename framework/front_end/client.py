# client.py
import requests
from urllib.parse import urljoin, quote

class AuthExpired(Exception):
    """Raised when the API returns 401 (token missing/expired)."""

# APIClient: simple wrapper around backend API to avoid repeating requests.get()/post()
class APIClient:
    def __init__(self, base: str):  # base URL
        self.base = base.rstrip("/")
        self.session = requests.Session()  # persistent session for connection reuse
        self.token = None

    # ---------- internals ----------
    def _headers(self):
        # Authorization header if logged in
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # Ensure path starts with '/' so urljoin behaves predictably
    def _norm(self, path: str) -> str:
        return path if path and path[0] == "/" else f"/{path}"

    def _handle(self, resp: requests.Response):
        if resp.status_code == 401:
            self.logout()
            raise AuthExpired("Authentication expired or invalid. Please log in again.")
        resp.raise_for_status()
        return resp

    def _get(self, path, **kwargs):
        # kwargs allow e.g. params={"id": 123}
        url = urljoin(self.base, self._norm(path))
        resp = requests.get(url, headers=self._headers(), timeout=60, **kwargs)
        return self._handle(resp)
    
    def _post(self, path, **kwargs):
        url = urljoin(self.base, self._norm(path))
        resp = requests.post(url, headers=self._headers(), timeout=120, **kwargs)
        return self._handle(resp)
    
    # ---------- auth ----------
    def is_authenticated(self) -> bool:
        return bool(self.token)

    def logout(self):
        self.token = None

    def register(self, username: str, password: str):
        r = self._post("/auth/register", json={"username": username, "password": password})
        return r.json()

    def login(self, username: str, password: str):
        # FastAPI’s OAuth2PasswordRequestForm expects form data, not JSON.
        r = self._post("/auth/token", data={"username": username, "password": password})
        # requests sets appropriate Content-Type for form data
        data = r.json()
        self.token = data.get("access_token")
        return data

    # ---------- model prediction ----------
    def predict(self, file_name: str, file_bytes: bytes, mime="image/png"):
        # Even if backend ignores filename/mime, multipart/form-data includes them.
        files = {"file": (file_name, file_bytes, mime)}
        r = self._post("/anomaly/predict", files=files)
        return r.json()
    
    def get_new_alerts(self):
        r = self._get("/anomaly/since")
        return r.json()
    
    def reset_alerts(self):
        r = self._post("/anomaly/reset")
        return r.json()

    # ---------- expert feedback / active learning ----------
    def get_pending_reviews(self, number: int):
        # GET /learning/review?limit=<number>
        r = self._get("/learning/review", params={"limit": number})
        data = r.json()
        return data.get("Records")

    def submit_expert_label(self, image_id: str, label: str):
        # POST /learning/label/{image_id} with body {"label": "..."}
        safe_id = quote(image_id, safe="")
        path = f"/learning/label/{safe_id}"
        r = self._post(path, json={"label": label})
        return r.json()

    def submit_classification_label(self, image_id: str, classification: str):
        # POST /learning/label/{image_id}/class with body {"classification": "..."}
        safe_id = quote(image_id, safe="")
        path = f"/learning/label/{safe_id}/class"
        r = self._post(path, json={"classification": classification})
        return r.json()
