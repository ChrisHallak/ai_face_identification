import logging
import urllib
from flask import Blueprint, jsonify, request
import time
import numpy as np
import os
import face_recognition
import io
import faiss
from image_utill import url_to_image

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Route works!"})

@api_blueprint.route('/image_to_vector', methods=['POST'])
def vectorize_image():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Missing or invalid JSON body"}), 400

        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "Field 'image_url' is required"}), 400

        # 1. Download image
        image_numpy = url_to_image(image_url)
        if image_numpy is None:
            return jsonify({"error": "The URL is not reachable or doesn't point to a valid image."}), 400

        # 2. Encode face
        input_encodings = face_recognition.face_encodings(image_numpy)
        if not input_encodings:
            return jsonify({"error": "No face detected in the input image"}), 400

        input_encoding = input_encodings[0]

        # 3. Return JSON-safe response
        return jsonify({
            "vector_length": len(input_encoding),
            "dtype": str(input_encoding.dtype),
            # "face_count": len(input_encodings),
            "vector": input_encoding.tolist()
        }), 200

    except Exception as e:
        logging.exception("Unhandled error in /vector endpoint")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_blueprint.route('/face_verify_local', methods=['POST'])
def verify_face_local():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        folder = data.get("folder")   # local directory for known faces
        file = data.get("file")       # network filename (URL path)
        base_url = data.get("base_url")  # base URL for remote file

        if not folder or not file or not base_url:
            return jsonify({"error": "Fields 'folder', 'file', and 'base_url' are required"}), 400

        if not os.path.exists(folder):
            return jsonify({"error": f"Folder '{folder}' does not exist"}), 400

        # --- Step 1: Download the input image from the network ---
        input_image_url = f"{base_url.rstrip('/')}/{file}"
        print(f"📥 Downloading input image from: {input_image_url}")

        try:
            with urllib.request.urlopen(input_image_url) as response:
                input_image_data = response.read()
        except urllib.error.URLError as e:
            return jsonify({"error": f"Failed to download input image: {str(e)}"}), 400

        # --- Step 2: Encode input image ---
        input_image = face_recognition.load_image_file(io.BytesIO(input_image_data))
        input_encodings = face_recognition.face_encodings(input_image)

        if not input_encodings:
            return jsonify({"error": "No face detected in the input image"}), 400

        input_encoding = input_encodings[0]

        # --- Step 3: Load known student faces from local folder ---
        known_encodings = []
        known_names = []
        failed = 0

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                student_image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(student_image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
                else:
                    failed += 1
            except Exception as e:
                print(f"⚠️ Failed to process {filename}: {str(e)}")
                failed += 1
                continue

        print(f"✅ Loaded {len(known_encodings)} student faces, failed: {failed}")

        if not known_encodings:
            return jsonify({"error": "No valid student faces found in the folder"}), 400

        # --- Step 4: Perform face matching ---
        tolerance = 0.45
        matches = face_recognition.compare_faces(known_encodings, input_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, input_encoding)

        best_match_index = np.argmin(face_distances)
        best_accuracy = 1 - face_distances[best_match_index]

        if matches[best_match_index]:
            student_name = known_names[best_match_index]
            print(f"✅ Match found: {student_name} (Confidence: {best_accuracy:.2%})")
            return jsonify({
                "student_name": student_name,
                "accuracy": round(best_accuracy * 100, 2),
                "error": None
            }), 200
        else:
            print("❌ No matching student found.")
            return jsonify({
                "student_name": None,
                "accuracy": round(best_accuracy * 100, 2),
                "error": "No matching student found"
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_blueprint.route('/db_embedding_local', methods=['POST'])
def db_embedding_local():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        folder_url = data.get("folder_url")

        if not folder_url:
            return jsonify({"error": "Field 'folder_url' is required"}), 400

        embeddings = []
        filenames = []
        failed = 0
        images_count = 0

        for filename in os.listdir(folder_url):
            path = os.path.join(folder_url, filename)
            images_count += 1

            if not os.path.isfile(path):
                continue
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                student_image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(student_image)
                if encodings:
                    # Convert numpy array to list for JSON serialization
                    encoding_list = encodings[0].tolist()
                    embeddings.append(encoding_list)
                    filenames.append(os.path.splitext(filename)[0])
                else:
                    failed += 1
            except Exception as e:
                print(f"⚠️ Failed to process {filename}: {str(e)}")
                failed += 1
                continue

        print(f"✅ Loaded {len(embeddings)} student faces, failed: {failed}")
        print(f'images_count: {images_count}')

        if not embeddings:
            return jsonify({"error": "No valid student faces found in the folder"}), 400

        return jsonify({
            "encodings": embeddings,
            "filenames": filenames,
            "stats": {
                "total_images_processed": images_count,
                "successful_embeddings": len(embeddings),
                "failed_processing": failed
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/face_match', methods=['POST'])
def face_match_using_face_recognition():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Missing or invalid JSON body"}), 400

        image_url = data.get("image_url")
        known_faces = data.get("known_faces")

        # Validation (same as before)
        if not image_url or not known_faces:
            return jsonify({"error": "Required fields missing"}), 400

        # 1. Precompute known encodings (convert once)
        ids = []
        encodings = []
        for item in known_faces:
            try:
                ids.append(item["id"])
                # Pre-allocate numpy array for better performance
                encodings.append(np.array(item["encoding"], dtype=np.float64, order='C'))
            except Exception:
                return jsonify({"error": "Invalid format in 'known_faces' list"}), 400

        # Convert to numpy array once
        encodings_array = np.array(encodings)

        # 2. Load image and get encoding
        image_numpy = url_to_image(image_url)
        if image_numpy is None:
            return jsonify({"error": "IMAGE_LOAD_FAILED"}), 400

        # 3. Use smaller model if acceptable (trade accuracy for speed)
        input_encodings = face_recognition.face_encodings(
            image_numpy,
            model="small"  # Uses 5 points instead of 68 (faster)
        )

        if not input_encodings:
            return jsonify({"error": "No face detected"}), 400

        input_encoding = input_encodings[0]

        # 4. Vectorized comparison (already efficient in library)
        tolerance = 0.5
        distances = face_recognition.face_distance(encodings_array, input_encoding)
        matches = distances <= tolerance  # More efficient than compare_faces

        # 5. Find best match
        if np.any(matches):
            best_index = np.argmin(distances)
            best_distance = distances[best_index]
            accuracy = max(0.0, 1 - best_distance)

            return jsonify({
                "id": ids[best_index],
                "accuracy": round(float(accuracy), 4)
            }), 200
        else:
            return jsonify({"message": "No matched face found"}), 404

    except Exception as e:
        print("Error in /face_match:", e)
        return jsonify({"error": "Internal server error"}), 500

@api_blueprint.route('/verify_face', methods=['POST'])
def match_using_faiss():

    try:
        start_time = time.time()
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Missing or invalid JSON body"}), 400

        image_url = data.get("image_url")
        known_faces = data.get("known_faces")

        # Validate input
        if not image_url or not known_faces:
            return jsonify({"error": "Required fields missing"}), 400

        # 1️. Prepare known encodings
        ids = []
        encodings = []
        for item in known_faces:
            try:
                ids.append(item["id"])
                encodings.append(np.array(item["encoding"], dtype=np.float32))
            except Exception:
                return jsonify({"error": "Invalid format in 'known_faces' list"}), 400

        encodings_array = np.array(encodings, dtype=np.float32)
        d = encodings_array.shape[1]  # dimension (usually 128)

        # 2️. Create FAISS index (L2 distance)
        start_indexing_time = time.time()
        index = faiss.IndexFlatL2(d)
        index.add(encodings_array)
        end_indexing_time = time.time()

        # 3️. Get input face encoding
        start_downloading_time = time.time()
        image_numpy = url_to_image(image_url)
        if image_numpy is None:
            return jsonify({"error": "IMAGE_LOAD_FAILED"}), 400
        end_downloading_time = time.time()

        start_encoding_image_time = time.time()
        input_encodings = face_recognition.face_encodings(image_numpy, model="small")
        if not input_encodings:
            return jsonify({"error": "No face detected"}), 400

        query = np.array([input_encodings[0]], dtype=np.float32)
        end_encoding_image_time = time.time()

        # 4️. Search for the nearest face
        start_searching_time = time.time()
        k = 1  # get top-1 nearest match
        distances, indices = index.search(query, k)
        end_searching_time = time.time()

        best_distance = float(distances[0][0])
        best_index = int(indices[0][0])

        # 5️. Decide if it’s a match based on threshold
        end_time = time.time()

        threshold = 0.3  # same logic as before
        if best_distance <= threshold:
            accuracy = max(0.0, 1 - best_distance)
            return jsonify({
                "id": ids[best_index],
                "accuracy": round(accuracy, 4),
                "distance": round(best_distance, 4),
                "downloading_time": round(end_downloading_time - start_downloading_time, 4),
                "searching_time": round(end_searching_time - start_searching_time, 4),
                "indexing_time": round(end_indexing_time - start_indexing_time, 4),
                "total_time": round(end_time - start_time, 4),
                "encoding_image_time": round(end_encoding_image_time - start_encoding_image_time, 4)

            }), 200
        else:
            return jsonify({"message": "No matched face found"}), 404


    except Exception as e:
        print("Error in /face_match:", e)
        return jsonify({"error": "Internal server error"}), 500