# pages/1_Supervised.py
import io  # in-memory bytes buffer
import os, glob

from urllib.parse import urljoin

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from PIL import Image

from client import AuthExpired

st.set_page_config(page_title="Anomaly Detection Pipeline", layout="wide")

st.title("Anomaly Detection Pipeline")

# Theme values (used sparsely for neutral blocks)
SECONDARY_BG = st.get_option("theme.secondaryBackgroundColor")
TEXT = st.get_option("theme.textColor")

# Ensure logged in and client exists
client = st.session_state.get("client")
if not client or not client.token:
    st.warning("Please login on the Home page first.")
    st.stop()

API_BASE = st.secrets.get("api_base", "http://127.0.0.1:8000")

# Ensure state holders are initialized
if "anomaly_records" not in st.session_state:
    st.session_state["anomaly_records"] = []                 


# Images live on disk
BASE_IMG_DIR = r"E:\ML - Self Study\MV4QC_back_end\images"

def resolve_image_path(username: str, image_id: str):
    # Locate file: <BASE_IMG_DIR>\<username>\<image_id>_*.jpg
    user_dir = os.path.join(BASE_IMG_DIR, username)
    pattern = os.path.join(user_dir, f"{image_id}_*.jpg")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

st.subheader("Live Anomaly Notifications")

colA, colB = st.columns([1, 1])
with colA:
    enable_live = st.checkbox("Enable live alerts", value=True, help="Poll the server for new anomaly alerts.")
with colB:
    poll_interval = st.slider("Poll interval (seconds)", 3, 30, 3, help="How often to check the server.")

alerts_panel = st.container()

def fetch_new_alerts():
    try:
        alerts = client.get_new_alerts() # list of new alerts
        st.session_state["anomaly_records"] = list(alerts)
    except AuthExpired:
        st.session_state.client.logout()
        st.rerun()
    except Exception as e:
        if "Unauthorized" in str(e):
            st.error("Session expired. Please login again.")
            st.session_state["client"] = None
        else: st.error(f"Failed to fetch anomalies: {e}")
        
if enable_live:
    st_autorefresh(interval=poll_interval * 1000, key="alerts_autorefresh")
    fetch_new_alerts()


alerts = list(st.session_state["anomaly_records"])
if "anomaly_amount" not in st.session_state or "prev_amount" not in st.session_state:
    st.session_state.anomaly_amount = 0
    st.session_state.prev_amount = 0

diff = len(alerts) - st.session_state.prev_amount
if diff < 0:
    diff = 0  # Don't allow negative counts
    
st.session_state.anomaly_amount += diff  # Update total anomaly count
st.session_state.prev_amount = len(alerts)  # Update previous count

if not alerts:
    st.info("No anomalies detected yet.")
else:
    cols = st.columns(2)

    with alerts_panel:
        st.write(f"Total anomalies detected: {len(alerts)}")

        # Helper: render one alert using mostly Streamlit primitives
        def render_alert_card(alert):
            job_id    = alert.get("job_id", "—")
            user      = alert.get("user", "—")
            label     = alert.get("label", "—")
            timestamp = alert.get("timestamp", "—")
            img_path  = resolve_image_path(user, job_id)

            # card container (we only style the outer div)
            with st.container():
                st.markdown('<div class="_alert_card">', unsafe_allow_html=True)

                # Header row: label (left) + timestamp (right)
                h1, h2 = st.columns([1, 1])
                with h1:
                    st.subheader("Anomaly Detected")
                with h2:
                    st.write(" ")
                    st.caption(f"{timestamp}")

                st.divider()

                # Details + image side-by-side
                c1, c2 = st.columns([1.2, 1.8])
                with c1:
                    st.write("**Image ID**")
                    st.code(job_id, language=None)
                    st.write("**User**")
                    st.code(user, language=None)
                with c2:
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Failed to open image: {e}")
                    else:
                        st.caption("No preview available")

                st.markdown("</div>", unsafe_allow_html=True)
        if len(alerts) == 1:
            # Single alert: render directly
            render_alert_card(alerts[0])
        else:
            left, right = st.columns(2, vertical_alignment="top")
            for i, alert in enumerate(alerts):
                with (left if i % 2 == 0 else right):
                    render_alert_card(alert) 
        
        st.divider()
        if st.button("I have checked all anomalies", type="primary", use_container_width=True):
            try:
                client.reset_alerts()
                st.session_state["anomaly_records"] = []
                st.success("Alerts reset successfully.")
            except AuthExpired:
                st.session_state.client.logout()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset alerts: {e}")


# Upload
st.subheader("Run a Prediction")

mode = st.radio(
    "Upload type",
    ["Multiple images", "ZIP folder"],
    index=0,
    help="Drag many images at once, or upload a .zip that contains images."
)
if mode == "Multiple images":
    uploads = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    zip_upload = None
else:
    uploads = []
    zip_upload = st.file_uploader(
        "Upload a ZIP containing images",
        type=["zip"],
        accept_multiple_files=False
    )

with st.expander("Upload options"):
    # Optional pre-processing for the image
    downscale = st.checkbox("Downscale large images before upload", value=True)
    max_side = st.slider(
        "Max side (px)", 512, 2048, 1280, step=64,
        help="Only used if downscale is enabled."
    )

def prepare_image_for_upload(pil_img: Image.Image) -> bytes:
    # PIL.Image (Image) -> byte stream (IO.BytesIO)
    # IO.BytesIO (bytes) -> Image (when reading back)
    img = pil_img.convert("RGB")
    if downscale:
        w, h = img.size
        if max(w, h) > max_side:
            img.thumbnail((max_side, max_side))  # keep aspect ratio
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def iter_images_from_ui(uploads, zip_upload):
    if uploads:
        for up in uploads:
            try:
                orig = Image.open(up).convert("RGB")
                yield up.name, orig
            except Exception as e:
                st.error(f"Could not read image {up.name}: {e}")
    elif zip_upload:
        import zipfile
        try:
            with zipfile.ZipFile(zip_upload) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".png", ".jpg", ".jpeg")) and not name.endswith("/"):
                        with zf.open(name) as file:
                            try:
                                orig = Image.open(file).convert("RGB")
                                yield os.path.basename(name), orig
                            except Exception as e:
                                st.error(f"Could not read image {name} in ZIP: {e}")
        except zipfile.BadZipFile:
            st.error("Uploaded file is not a valid ZIP archive.")


has_input = (uploads and len(uploads) > 0) or (zip_upload is not None)
if not has_input:
    st.info("Please upload images or a ZIP file to run predictions.")
    st.stop()   
# Prediction by upload
else:
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Connecting to the Prediction Server"):
            results = []
            errors = []
            
            all_items = list(iter_images_from_ui(uploads, zip_upload))
            total = len(all_items)
            if total == 0:
                st.warning("No valid images found in the upload.")
                st.stop()

            for i, (name, orig) in enumerate(all_items):
                with st.spinner(f"Uploading {name} ({i+1}/{total})"):
                    try:
                        payload = prepare_image_for_upload(orig)
                        response = client.predict(name, payload, mime="image/png")
                        results.append({"file":name, "response":response})
                    except AuthExpired:
                        st.session_state.client.logout()
                        st.rerun()
                    except Exception as e:
                        errors.append({"file": name, "error": str(e)})
                        st.stop()

        st.success(f"Done. Processed {len(results)} of {total} file(s).")
        if results:
            st.write("Prediction Queue Status (per file)")
            st.json(results)
        if errors:
            st.warning("Some files could not be processed:")
            st.json(errors)
