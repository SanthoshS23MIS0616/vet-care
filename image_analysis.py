import cv2
import numpy as np

def analyze_skin_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize for faster processing
    image = cv2.resize(image, (300, 300))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect redness
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    redness_ratio = np.sum(mask1 > 0) / (300 * 300)

    # Brightness
    brightness = np.mean(hsv[:, :, 2])

    # Texture roughness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (300 * 300)

    report = f"""
    Image Analysis Report:
    - Redness level: {redness_ratio:.2f}
    - Brightness level: {brightness:.2f}
    - Skin roughness level: {edge_ratio:.2f}
    """

    return report