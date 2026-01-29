# pages/3_Expert_Review.py
import os, glob
import streamlit as st
from PIL import Image

from client import AuthExpired

st.set_page_config(page_title="Expert Review", layout="wide")

st.title("Review — Pending Records")

# Theme values for small neutral surfaces / captions if needed
SECONDARY_BG = st.get_option("theme.secondaryBackgroundColor")
TEXT = st.get_option("theme.textColor")

client = st.session_state.get("client")
if not client or not client.token:
    st.warning("Please login on the Home page first.")
    st.stop()
    # st.session_state is a persistent dict per user session.
    # - Regular Python variables reset each rerun.
    # - st.session_state persists across reruns.

# Ensure state holders are initialized
if "records" not in st.session_state:
    st.session_state["records"] = []               # Latest fetched list from backend
if "review_state" not in st.session_state:
    st.session_state["review_state"] = {}          # Per image_id: {"labeled": bool, "label": str}
if "active_record" not in st.session_state:
    st.session_state["active_record"] = None       # Currently opened record for popup panel
if "active_img_path" not in st.session_state:
    st.session_state["active_img_path"] = None

# Images live on disk
BASE_IMG_DIR = r"E:\ML - Self Study\MV4QC_back_end\images"

def resolve_image_path(username: str, image_id: str):
    # Locate file: <BASE_IMG_DIR>\<username>\<image_id>_*.jpg
    user_dir = os.path.join(BASE_IMG_DIR, username)
    pattern = os.path.join(user_dir, f"{image_id}_*.jpg")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

# Reload Records for Expert Review
with st.form(key="fetch_from_AL"):
    # Slider widget — user picks a number
    number_value = st.slider(
        "Select a number",
        min_value=1, max_value=15, value=10, step=1,
        key="slider_for_fetch"
    )
    reload_images = st.form_submit_button("Reload from Active Learning Module")

if reload_images:
    try:
        # Fetch anomaly records selected by AL
        records = client.get_pending_reviews(number_value)
        # Fall back if client returns dict
        if isinstance(records, dict):
            records = records.get("Records", [])
        st.session_state["records"] = list(records)
        st.success(f"Loaded {len(records)} records.")
    except AuthExpired:
        st.session_state.client.logout()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to fetch pending records: {e}")

# Notes on records state handling:
# - If page is refreshed (not reloaded via the form), we keep previous st.session_state["records"].
# - If reloaded via the form, st.session_state["records"] is updated from backend.

# Establish working list
records = list(st.session_state["records"])

if not records:
    st.info("No records are found for expert review.")
    st.stop()

# ==============================================================
# POP-UP Panel — shown when a card's "Open" is clicked
# ==============================================================
if st.session_state["active_record"] is not None:
    # Passed from the grid card
    rec = st.session_state["active_record"]
    image_id = rec.get("id")
    username = rec.get("user")

    st.markdown("---")
    st.subheader("Review Image")
    c1, c2 = st.columns([3, 2])

    with c1:
        if st.session_state["active_img_path"]:
            try:
                img = Image.open(st.session_state["active_img_path"]).convert("RGB")
                st.image(img, caption=os.path.basename(st.session_state["active_img_path"]), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to open image: {e}")
        else:
            st.warning("Image file route is not found on the disk.")

    # We keep a per-record global state map via keys generated from image_id.
    # Each time we enter this page and fetch a batch, records initialize with default states.
    # Re-labeling is possible via re-initialization.
    # state = st.session_state["review_state"]
    with c2:
        st.write(f"Uploaded by {username}")
        st.write(f"Image ID: {image_id}")

        state_key = f"review_{image_id}"
        state = st.session_state["review_state"].setdefault(
            state_key, {"labeled": False, "label": None}
        )
        st.write(state["labeled"], state["label"])

        # Submit Label
        if not state["labeled"]:
            # Unique form per record (separates forms across cards)
            with st.form(key=f"form_label_{image_id}"):
                # Unique widget storage per record
                label_choice = st.selectbox(
                    "Expert label",
                    ["Anomalous", "Normal"],
                    key=f"label_{image_id}"
                )
                submit_label = st.form_submit_button("Submit Label")

            if submit_label:
                try:
                    mapped = "anomalous" if label_choice == "Anomalous" else "normal"
                    response = client.submit_expert_label(image_id, mapped)
                    st.success("Expert label submitted.")
                    st.json(response)
                    # Prepare for classification step
                    state["labeled"] = True
                    state["label"] = mapped
                    st.rerun()  # refresh the page to show the next step if desired
                except AuthExpired:
                    st.session_state.client.logout()
                    st.rerun()
                except Exception as e:
                    st.error(f"Submit failed: {e}")

            st.caption("Set a Class for the Anomalous Record")

            if state["labeled"] and state["label"] == "anomalous":
                with st.form(key=f"form_class_{image_id}"):
                    classification = st.text_input(
                        "Final classification",
                        key=f"class_{image_id}"
                    )
                    submit_class = st.form_submit_button("Submit Class")
                
                st.write(st.session_state["review_state"][state_key]["label"])
                if submit_class:
                    if not classification.strip():
                        st.warning("Please provide a classification")
                    else:
                        try:
                            st.write("HERE")
                            response = client.submit_classification_label(image_id, classification.strip())
                            st.success("Classification submitted.")
                            st.json("response")
                        except AuthExpired:
                            st.session_state.client.logout()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Submit failed: {e}")
            elif state["labeled"] and state["label"] == "normal":
                st.info('No classification required for "Normal" items.')

        # If already labeled, unlock class submission (for anomalous)
        else:
            st.caption("Set a Class for the Anomalous Record")

            if st.session_state["review_state"][state_key]["label"] == "anomalous":
                with st.form(key=f"form_class_{image_id}"):
                    classification = st.text_input(
                        "Final classification",
                        key=f"class_{image_id}"
                    )
                    submit_class = st.form_submit_button("Submit Class")

                if submit_class:
                    if not classification.strip():
                        st.warning("Please provide a classification")
                    else:
                        try:
                            response = client.submit_classification_label(image_id, classification.strip())
                            st.success("Classification submitted.")
                            st.json(response)
                        except Exception as e:
                            st.error(f"Submit failed: {e}")
            else:
                st.info('No classification required for "Normal" items.')

        # Close / back
        if st.button("Close", type="secondary"):
            st.session_state["active_record"] = None
            st.session_state["active_img_path"] = None
            st.rerun()

    st.markdown("---")

# ==============================================================
# GRID OF CARDS
# ==============================================================
cols = st.columns(3)

for i, rec in enumerate(records):
    # Enumerate -> i index; rec is an anomaly record
    # Common names: "id" or "image_id"; "username" or "user"
    image_id = rec.get("id")
    username = rec.get("user")

    if not image_id or not username:
        cols[i % 3].warning(f"Record missing username/image_id: {rec}")
        continue

    img_path = resolve_image_path(username, image_id)

    with cols[i % 3]:
        st.write(f"Uploaded by {username}")
        st.write(f"Image ID: {image_id}")

        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to open image: {e}")
        else:
            st.warning("Image file not found on disk.")

        # Open detail panel (simulated 'clickable image')
        if st.button("Open", key=f"open_{image_id}", use_container_width=True):
            state_key = f"review_{image_id}"
            st.session_state["review_state"][state_key] = {"labeled": False, "label": None}

            st.session_state["active_record"] = rec
            st.session_state["active_img_path"] = img_path
            st.rerun()

        # Optional: raw record for debugging
        with st.expander("Record data"):
            st.json(rec)
