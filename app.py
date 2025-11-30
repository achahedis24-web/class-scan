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

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
# Download your service account key from Firebase Console
# and save it as 'serviceAccountKey.json' in the flask-api folder
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://classscan-4fc7a-default-rtdb.europe-west1.firebasedatabase.app',
    'storageBucket': 'classscan-4fc7a.firebasestorage.app'
})

def load_image_from_url(url):
    """Load image from Firebase Storage URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def process_student_image(image_url):
    """Process a single student image and extract face encodings"""
    try:
        student_image = load_image_from_url(image_url)
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
        "imageUrl": "https://firebase-storage-url..."
    }
    
    Returns:
    {
        "encodings": [array of floats],
        "success": true/false
    }
    """
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        
        if not image_url:
            return jsonify({'error': 'Missing imageUrl'}), 400
        
        encodings = process_student_image(image_url)
        
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
        "classImageUrl": "https://firebase-storage-url...",
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
        class_image_url = data.get('classImageUrl')
        students = data.get('students', [])
        
        if not class_image_url:
            return jsonify({'error': 'Missing classImageUrl'}), 400
        
        if not students:
            return jsonify({'error': 'Missing students data'}), 400
        
        # Process the attendance
        results = process_class_image(class_image_url, students)
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
