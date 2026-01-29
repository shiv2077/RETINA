from PIL import Image
import io
import os

buffer = {
    "count": 0,
    "max_size": 1,  # run unsupervised every N jobs
    "dict_list": {}
}

def should_run_unsupervised(image_id: str, username: str, image_bytes: bytes) -> bool:
    """
    Check if the unsupervised anomaly detection should be run based on the buffer count
    """

    buffer["count"] += 1
    buffer["dict_list"][image_id] = (username, image_bytes) # tuple
    if buffer["count"] >= buffer["max_size"]:
        return True
    return False

def reset_buffer():
    """
    Reset the buffer count and clear the list
    """
    buffer["count"] = 0
    buffer["dict_list"] = {}

def get_buffer_list():
    """
    Get the current buffer list of images
    """
    return buffer["dict_list"]

def get_buffer_capacity():
    """
    Get the current buffer capacity
    """
    return buffer["max_size"]