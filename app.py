import os
import re
import cv2
import sqlite3
import requests
import datetime
import threading
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, g
from ultralytics import YOLO

# CONFIGURATION
class Config:
    SECRET_KEY = 'change_this_to_secure_random_key'
    MODEL_PATH = 'model/my_model.pt'
    OCR_API_KEY = 'K86149120588957'
    DB_NAME = "gate_system.db"
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    CONFIDENCE_THRESHOLD = 0.60
    CROP_WIDTH_RATIO = 0.85 

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global thread-safe variable for gate hardware state
gate_state = {"is_open": False}

# DATABASE HANDLERS
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DB_NAME'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database schema."""
    with app.app_context():
        db = get_db()
        # Employee Table
        db.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                position TEXT NOT NULL
            )
        """)
        # License Plates
        db.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                plate_number TEXT PRIMARY KEY,
                employee_id INTEGER,
                FOREIGN KEY (employee_id) REFERENCES employees (id) ON DELETE CASCADE
            )
        """)
        # Access Logs
        db.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER,
                plate_number TEXT,
                action TEXT DEFAULT 'ENTRY', 
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        """)
        db.commit()

# AI & VISION PIPELINE
model = None
def get_yolo_model():
    global model
    if model is None and os.path.exists(Config.MODEL_PATH):
        try:
            model = YOLO(Config.MODEL_PATH)
        except Exception as e:
            print(f"[Error] Model Load Failed: {e}")
    return model

def ocr_plate_text(image_path):
    """Sends image to OCR.Space API."""
    payload = {
        'apikey': Config.OCR_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False,
        'detectOrientation': True,
        'scale': True,
        'OCREngine': '2'
    }
    try:
        with open(image_path, 'rb') as f:
            r = requests.post('https://api.ocr.space/parse/image', 
                            files={'filename': f}, data=payload, timeout=8)
        result = r.json()
        
        if not result.get('IsErroredOnProcessing') and result.get('ParsedResults'):
            raw_text = result['ParsedResults'][0].get('ParsedText', "")
            return re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    except Exception as e:
        print(f"[Error] OCR API Failed: {e}")
    return None

def process_upload(file_storage):
    """
    1. Saves upload
    2. Runs YOLO Detection
    3. Crops Plate
    4. Runs OCR
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"full_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)

    active_model = get_yolo_model()
    if not active_model:
        return None, filename, None

    img = cv2.imread(filepath)
    if img is None: return None, filename, None

    results = active_model(img, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
    
    plate_text = None
    crop_filename = None

    # Save annotated main image (with boxes)
    annotated_img = results[0].plot()
    cv2.imwrite(filepath, annotated_img) 

    # Iterate detections
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Crop logic
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            # Refine crop (remove country strip)
            h, w = plate_crop.shape[:2]
            plate_crop = plate_crop[:, :int(w * Config.CROP_WIDTH_RATIO)]

            # Save crop
            crop_name = f"crop_{timestamp}.jpg"
            crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_name)
            cv2.imwrite(crop_path, plate_crop)
            crop_filename = crop_name

            # OCR
            detected = ocr_plate_text(crop_path)
            if detected and len(detected) > 2: # Filter noise
                plate_text = detected
                break 
        if plate_text: break

    return plate_text, filename, crop_filename

# GATE HARDWARE LOGIC
def _auto_close_gate():
    """Background task to close gate after delay (Now 10 seconds)."""
    # Changed from 5 to 10 seconds
    time.sleep(10) 
    gate_state["is_open"] = False

def activate_gate():
    """Opens gate and schedules auto-close thread."""
    gate_state["is_open"] = True
    threading.Thread(target=_auto_close_gate, daemon=True).start()


# ROUTES
# --- Auth ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin123':
            session['user'] = 'admin'
            return redirect(url_for('detection'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- Main Operation ---
@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'user' not in session: return redirect(url_for('login'))
    
    # Context initialization prevents VariableUndefined errors
    context = {
        'text': None,
        'full_image': None,
        'plate_image': None,
        'status': None,
        'message': None,
        'css_class': '',
        'gate_triggered': False
    }
    
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            text, full_img, crop_img = process_upload(file)
            
            # Default Fail State
            context.update({
                'full_image': full_img,
                'plate_image': crop_img,
                'status': "FAILED",
                'message': "No Plate Detected",
                'css_class': "status-denied"
            })

            if text:
                db = get_db()
                # Check DB for plate
                employee = db.execute("""
                    SELECT e.name, p.employee_id 
                    FROM plates p 
                    JOIN employees e ON p.employee_id = e.id 
                    WHERE p.plate_number = ?
                """, (text,)).fetchone()

                if employee:
                    activate_gate() # Physical Trigger
                    context.update({
                        'text': text,
                        'status': "GRANTED",
                        'message': f"Authorized: {employee['name']}",
                        'css_class': "status-granted",
                        'gate_triggered': True
                    })
                    # Log Entry
                    db.execute("INSERT INTO access_logs (employee_id, plate_number, details) VALUES (?, ?, ?)",
                               (employee['employee_id'], text, "Automated Entry"))
                else:
                    context.update({
                        'text': text,
                        'status': "DENIED",
                        'message': "Unregistered Vehicle",
                        'css_class': "status-denied"
                    })
                    # Log Denial
                    db.execute("INSERT INTO access_logs (plate_number, action, details) VALUES (?, ?, ?)",
                               (text, "DENIED", "Unknown Vehicle"))
                db.commit()

    return render_template('detection.html', **context)

# --- Admin Dashboard ---
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    db = get_db()

    if request.method == 'POST':
        # Add Employee
        if 'add_employee' in request.form:
            name = request.form['name']
            pos = request.form['position']
            plate = re.sub(r'[^A-Z0-9]', '', request.form['plate'].upper())
            
            cur = db.cursor()
            cur.execute("INSERT INTO employees (name, position) VALUES (?, ?)", (name, pos))
            eid = cur.lastrowid
            if plate:
                cur.execute("INSERT INTO plates (plate_number, employee_id) VALUES (?, ?)", (plate, eid))
            db.commit()
            
        # Delete Employee
        elif 'delete_id' in request.form:
            db.execute("DELETE FROM employees WHERE id = ?", (request.form['delete_id'],))
            db.execute("DELETE FROM plates WHERE employee_id = ?", (request.form['delete_id'],))
            db.commit()
            return redirect(url_for('dashboard'))

    # Fetch Logs
    logs = db.execute("""
        SELECT a.timestamp, IFNULL(e.name, 'Visitor') as name, a.plate_number, a.action 
        FROM access_logs a 
        LEFT JOIN employees e ON a.employee_id = e.id 
        ORDER BY a.timestamp DESC LIMIT 20
    """).fetchall()

    # Fetch Employees
    employees = db.execute("""
        SELECT e.id, e.name, e.position, p.plate_number 
        FROM employees e 
        LEFT JOIN plates p ON e.id = p.employee_id
    """).fetchall()

    return render_template('dashboard.html', logs=logs, employees=employees)

# --- Gate API ---
@app.route('/gate')
def gate_view():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('gate.html')

@app.route('/api/gate_status')
def api_gate_status():
    return jsonify(gate_state)

@app.route('/api/trigger_gate', methods=['POST'])
def api_trigger_gate():
    if 'user' in session:
        activate_gate()
        return jsonify({'status': 'opened'})
    return jsonify({'status': 'error'}), 401

# --- Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(Config.DB_NAME):
        init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)