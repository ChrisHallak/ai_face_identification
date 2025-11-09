
import io
import urllib
import face_recognition

def url_to_image(url: str):
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None
