from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db, storage
import os
import base64
import json

app = Flask(__name__)
# Limit incoming request body size to 20MB to avoid memory issues with very large uploads
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
# Configure CORS to explicitly allow the JSON Content-Type header and all origins in dev
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type"]) 

# Initialize Firebase Admin SDK
# In production (Render), use environment variables
# In development, fall back to local serviceAccountKey.json
def get_firebase_credentials():
    """Load Firebase credentials from environment variables or local file"""
    # Try to load from environment variables (production)
    if all(os.getenv(key) for key in ['TYPE', 'PROJECT_ID', 'PRIVATE_KEY_ID', 'PRIVATE_KEY', 'CLIENT_EMAIL', 'CLIENT_ID', 'AUTH_URI', 'TOKEN_URI', 'AUTH_PROVIDER_X509_CERT_URL', 'CLIENT_X509_CERT_URL', 'UNIVERSE_DOMAIN']):
        cred_dict = {
            'type': os.getenv('TYPE'),
            'project_id': os.getenv('PROJECT_ID'),
            'private_key_id': os.getenv('PRIVATE_KEY_ID'),
            'private_key': os.getenv('PRIVATE_KEY').replace('\\n', '\n'),  # Handle escaped newlines
            'client_email': os.getenv('CLIENT_EMAIL'),
            'client_id': os.getenv('CLIENT_ID'),
            'auth_uri': os.getenv('AUTH_URI'),
            'token_uri': os.getenv('TOKEN_URI'),
            'auth_provider_x509_cert_url': os.getenv('AUTH_PROVIDER_X509_CERT_URL'),
            'client_x509_cert_url': os.getenv('CLIENT_X509_CERT_URL'),
            'universe_domain': os.getenv('UNIVERSE_DOMAIN')
        }
        return credentials.Certificate(cred_dict)
    # Fall back to local file (development)
    elif os.path.exists('serviceAccountKey.json'):
        return credentials.Certificate('serviceAccountKey.json')
    else:
        raise ValueError('Firebase credentials not found. Set environment variables or provide serviceAccountKey.json')

cred = get_firebase_credentials()
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://classscan-4fc7a-default-rtdb.europe-west1.firebasedatabase.app',
    'storageBucket': 'classscan-4fc7a.firebasestorage.app'
})

def load_image_from_url(url):
    """Load image from Firebase Storage URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def process_student_image_from_base64(image_data):
    """Process a student image from base64 data and extract face encodings"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            # Remove data URL prefix (e.g., 'data:image/jpeg;base64,')
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        student_image = np.array(image)
        
        encodings = face_recognition.face_encodings(student_image)
        
        if len(encodings) == 0:
            return None
        
        return encodings[0].tolist()
    except Exception as e:
        print(f"Error processing student image: {str(e)}")
        return None

def process_class_image(class_image_url, students_data):
    """
    Process class photo and detect which students are present/absent
    
    Args:
        class_image_url: URL of the class photo
        students_data: List of dicts with 'id', 'name', and 'encodings'
    
    Returns:
        List of dicts with 'id', 'name', and 'isAbsent' boolean
    """
    try:
        # Load class image
        class_img = load_image_from_url(class_image_url)
        class_encodings = face_recognition.face_encodings(class_img)
        
        # Initialize all students as absent
        results = []
        for student in students_data:
            results.append({
                'id': student['id'],
                'name': student['name'],
                'isAbsent': True
            })
        
        # If no faces detected in class photo, all are absent
        if len(class_encodings) == 0:
            return results
        
        # Compare each face in class photo with student encodings
        for face_encoding in class_encodings:
            for i, student in enumerate(students_data):
                if not student.get('encodings'):
                    continue
                
                # Convert encodings back to numpy array
                student_encoding = np.array(student['encodings'])
                
                # Compare faces (threshold of 0.6 for matching)
                matches = face_recognition.compare_faces(
                    [student_encoding], 
                    face_encoding,
                    tolerance=0.6
                )
                
                if matches[0]:
                    results[i]['isAbsent'] = False
        
        return results
    
    except Exception as e:
        print(f"Error processing class image: {str(e)}")
        raise

def process_class_image_from_base64(image_data, students_data):
    """
    Process class photo from base64 data and detect which students are present/absent
    
    Args:
        image_data: Base64 encoded image data
        students_data: List of dicts with 'id', 'name', and 'encodings'
    
    Returns:
        List of dicts with 'id', 'name', and 'isAbsent' boolean
    """
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            # Remove data URL prefix (e.g., 'data:image/jpeg;base64,')
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        class_img = np.array(image)
        
        class_encodings = face_recognition.face_encodings(class_img)
        print(f"Found {len(class_encodings)} faces in class photo")
        
        # Initialize all students as absent
        results = []
        for student in students_data:
            results.append({
                'id': student['id'],
                'name': student['name'],
                'isAbsent': True
            })
        
        print(f"Processing {len(students_data)} students")
        
        # If no faces detected in class photo, all are absent
        if len(class_encodings) == 0:
            print("No faces detected in class photo - all students marked absent")
            return results
        
        # Compare each face in class photo with student encodings
        for face_idx, face_encoding in enumerate(class_encodings):
            print(f"\nComparing detected face {face_idx + 1}...")
            for i, student in enumerate(students_data):
                if not student.get('encodings'):
                    print(f"  Student {student['name']}: No encoding available")
                    continue
                
                # Convert encodings back to numpy array
                student_encoding = np.array(student['encodings'])
                
                # Compare faces (threshold of 0.6 for matching)
                matches = face_recognition.compare_faces(
                    [student_encoding], 
                    face_encoding,
                    tolerance=0.6
                )
                
                if matches[0]:
                    print(f"  ✓ MATCH: Student {student['name']} is PRESENT")
                    results[i]['isAbsent'] = False
        
        print(f"\nFinal results: {results}")
        return results
    
    except Exception as e:
        print(f"Error processing class image from base64: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/encode-student', methods=['POST'])
def encode_student():
    """
    Encode a student's face from their reference photo
    
    Expected JSON:
    {
        "imageData": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
    
    Returns:
    {
        "encodings": [array of floats],
        "success": true/false
    }
    """
    try:
        data = request.get_json()
        image_data = data.get('imageData')
        
        if not image_data:
            return jsonify({'error': 'Missing imageData'}), 400
        
        encodings = process_student_image_from_base64(image_data)
        
        if encodings is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in image'
            }), 400
        
        return jsonify({
            'success': True,
            'encodings': encodings
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process-attendance', methods=['POST'])
def process_attendance():
    """
    Process attendance from a class photo
    
    Expected JSON:
    {
        "classImageData": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
        "students": [
            {
                "id": "student-id",
                "name": "Student Name",
                "encodings": [array of floats]
            },
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "results": [
            {
                "id": "student-id",
                "name": "Student Name",
                "isAbsent": true/false
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        class_image_data = data.get('classImageData')
        students = data.get('students', [])
        
        print(f"Received data keys: {data.keys() if data else 'None'}")
        print(f"Image data length: {len(class_image_data) if class_image_data else 'None'}")
        print(f"Number of students: {len(students)}")
        
        if not class_image_data:
            return jsonify({'error': 'Missing classImageData', 'success': False}), 400
        
        if not students:
            return jsonify({'error': 'Missing students data', 'success': False}), 400
        
        # Process the attendance from base64 image
        results = process_class_image_from_base64(class_image_data, students)
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        print(f"Exception in process_attendance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
