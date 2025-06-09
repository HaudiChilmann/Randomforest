from flask import Flask, render_template, jsonify, request
import firebase_admin
from firebase_admin import credentials, db, firestore
import os
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import atexit
import json

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi Firebase dari Environment Variable
cred_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')

if not cred_json_str:
    # Ini akan membuat aplikasi crash jika variabel tidak ditemukan,
    # yang bagus untuk mengetahui error lebih awal.
    raise ValueError("Variabel GOOGLE_CREDENTIALS_JSON tidak diatur di Render.")

cred_info = json.loads(cred_json_str)
cred = credentials.Certificate(cred_info)

# Cek apakah aplikasi Firebase sudah diinisialisasi atau belum
# Ini untuk mencegah error jika kode ini terpanggil lebih dari sekali
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://anggur-dataset-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# Inisialisasi Firestore
firestore_client = firestore.client()

# Load model Random Forest
model_path = os.path.join(os.path.dirname(__file__), 'model_rf')
try:
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    print("Model Random Forest berhasil dimuat")
except Exception as e:
    print(f"Error loading model: {e}")
    rf_model = None

def check_watering_conditions(temperature, humidity, soil_moisture):
    """
    Fungsi untuk menentukan keputusan penyiraman berdasarkan ambang batas:
    - Suhu udara: 20°C-31°C
    - Kelembapan udara: 75-80%
    - Kelembapan tanah: 60-75%
    
    Return 1 untuk "siram", 0 untuk "jangan siram"
    """
    # Cek apakah parameter berada di luar rentang optimal
    temp_out_of_range = temperature < 20 or temperature > 31
    humidity_out_of_range = humidity < 75 or humidity > 80
    soil_out_of_range = soil_moisture < 60 or soil_moisture > 75
    
    # Jika salah satu parameter di luar rentang, maka siram (return 1)
    if temp_out_of_range or humidity_out_of_range or soil_out_of_range:
        return 1  # Siram
    else:
        return 0  # Jangan siram

def parse_datetime_string(datetime_str):
    """
    Fungsi untuk mengparse string DateTime menjadi timestamp Unix.
    Mendukung berbagai format DateTime dengan penanganan yang lebih baik.
    """
    try:
        if not datetime_str or datetime_str == '':
            logger.warning("Empty datetime string provided")
            return None
            
        # Bersihkan string dari whitespace
        datetime_str = str(datetime_str).strip()
        
        # Format yang mungkin ada di Firestore
        formats = [
            "%Y-%m-%d %H:%M:%S",      # 2025-06-04 21:15:15
            "%Y-%m-%dT%H:%M:%S",      # 2025-06-04T21:15:15
            "%Y-%m-%d %H:%M:%S.%f",   # 2025-06-04 21:15:15.123456
            "%Y-%m-%dT%H:%M:%S.%f",   # 2025-06-04T21:15:15.123456
            "%Y-%m-%dT%H:%M:%S.%fZ",  # 2025-06-04T21:15:15.123456Z
            "%Y-%m-%dT%H:%M:%SZ",     # 2025-06-04T21:15:15Z
            "%d/%m/%Y %H:%M:%S",      # 04/06/2025 21:15:15
            "%d-%m-%Y %H:%M:%S",      # 04-06-2025 21:15:15
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                timestamp = dt.timestamp()
                logger.info(f"Successfully parsed '{datetime_str}' with format '{fmt}' -> timestamp: {timestamp}")
                return timestamp
            except ValueError:
                continue
                
        # Jika semua format gagal, coba parsing ISO format
        try:
            # Handle Z timezone
            if datetime_str.endswith('Z'):
                datetime_str = datetime_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(datetime_str)
            timestamp = dt.timestamp()
            logger.info(f"Successfully parsed '{datetime_str}' with ISO format -> timestamp: {timestamp}")
            return timestamp
        except ValueError:
            pass
            
        logger.error(f"Could not parse datetime string: '{datetime_str}'")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing datetime string '{datetime_str}': {e}")
        return None

def normalize_timestamp(timestamp_value):
    """
    Fungsi untuk normalisasi timestamp dari berbagai format dengan validasi yang lebih ketat.
    """
    try:
        if timestamp_value is None or timestamp_value == '':
            logger.warning("Empty timestamp value, using current time")
            return datetime.now().timestamp()
        
        # Jika sudah berupa number (int/float)
        if isinstance(timestamp_value, (int, float)):
            # Validasi range timestamp yang masuk akal
            # Timestamp untuk 1 Jan 2020 = 1577836800
            # Timestamp untuk 1 Jan 2030 = 1893456000
            if 1577836800 <= timestamp_value <= 1893456000:
                logger.info(f"Valid timestamp in seconds: {timestamp_value}")
                return float(timestamp_value)
            elif 1577836800000 <= timestamp_value <= 1893456000000:
                # Timestamp dalam milliseconds
                result = timestamp_value / 1000
                logger.info(f"Converted milliseconds timestamp {timestamp_value} to seconds: {result}")
                return result
            else:
                logger.warning(f"Timestamp {timestamp_value} is out of valid range, using current time")
                return datetime.now().timestamp()
        
        # Jika berupa string
        if isinstance(timestamp_value, str):
            # Coba parse sebagai number terlebih dahulu
            try:
                num_timestamp = float(timestamp_value)
                return normalize_timestamp(num_timestamp)  # Rekursi untuk validasi
            except ValueError:
                # Jika bukan number, coba parse sebagai datetime string
                parsed_timestamp = parse_datetime_string(timestamp_value)
                if parsed_timestamp:
                    return parsed_timestamp
        
        # Fallback ke timestamp sekarang
        logger.warning(f"Using current timestamp as fallback for: {timestamp_value}")
        return datetime.now().timestamp()
        
    except Exception as e:
        logger.error(f"Error normalizing timestamp {timestamp_value}: {e}")
        return datetime.now().timestamp()

def get_detailed_sort_key(item):
    """
    Fungsi untuk membuat sort key yang detail berdasarkan:
    1. Tahun
    2. Bulan  
    3. Tanggal
    4. Jam
    5. Menit
    6. Detik
    """
    try:
        timestamp = item.get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp)
        # Return tuple untuk sorting: (tahun, bulan, tanggal, jam, menit, detik)
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    except Exception as e:
        logger.error(f"Error creating sort key for timestamp {timestamp}: {e}")
        # Fallback ke timestamp asli jika ada error
        return (1970, 1, 1, 0, 0, 0)

@app.route('/api/sensor-data')
def get_sensor_data():
    """
    Endpoint API untuk mendapatkan data sensor dari Firebase Realtime Database dan Firestore.
    Dengan perbaikan parsing timestamp yang lebih akurat.
    """
    try:
        # Ambil parameter dari query string
        start_param = request.args.get('start', type=int)
        end_param = request.args.get('end', type=int)
        limit_param = request.args.get('limit', default=100, type=int)
        sort_method = request.args.get('sort', default='datetime', type=str)
        
        logger.info(f"API sensor-data called with start={start_param}, end={end_param}, limit={limit_param}, sort={sort_method}")
        
        # Ambil data dari Firestore
        collection_ref = firestore_client.collection('sensor_history')
        
        # Jika ada parameter start dan end, filter berdasarkan timestamp
        if start_param and end_param:
            logger.info(f"Filtering data from {datetime.fromtimestamp(start_param).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_param).strftime('%Y-%m-%d %H:%M:%S')}")
            query = collection_ref.where('timestamp', '>=', start_param).where('timestamp', '<=', end_param)
            docs = query.order_by('timestamp').limit(limit_param * 2).stream()
        else:
            # Jika tidak ada parameter waktu, ambil data terbaru
            logger.info(f"Getting latest {limit_param} records")
            docs = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit_param * 2).stream()

        # Kumpulkan data dari Firestore
        firestore_data = []
        for doc in docs:
            data = doc.to_dict()
            
            # Ekstrak data sensor
            humidity_firestore = data.get('humidity', 0)
            temperature_firestore = data.get('temperature', 0)
            soil_moisture_firestore = data.get('soil_moisture', 0)
            
            # Ekstrak timestamp - PRIORITASKAN DateTime field karena lebih akurat
            datetime_firestore = data.get('DateTime')
            timestamp_firestore = data.get('timestamp')
            
            # Normalisasi timestamp dengan prioritas DateTime field
            normalized_timestamp = None
            
            if datetime_firestore:
                logger.info(f"Processing DateTime field: {datetime_firestore}")
                normalized_timestamp = parse_datetime_string(datetime_firestore)
                if normalized_timestamp:
                    logger.info(f"Successfully parsed DateTime: {datetime_firestore} -> {normalized_timestamp}")
                else:
                    logger.warning(f"Failed to parse DateTime: {datetime_firestore}")
            
            # Jika DateTime gagal atau tidak ada, coba timestamp field
            if not normalized_timestamp and timestamp_firestore:
                logger.info(f"Fallback to timestamp field: {timestamp_firestore}")
                normalized_timestamp = normalize_timestamp(timestamp_firestore)
            
            # Jika kedua field gagal, gunakan waktu sekarang
            if not normalized_timestamp:
                logger.warning(f"No valid timestamp found in document {doc.id}, using current time")
                normalized_timestamp = datetime.now().timestamp()
            
            # Validasi timestamp final (harus dalam rentang yang masuk akal)
            current_time = datetime.now().timestamp()
            if normalized_timestamp < 1577836800:  # Sebelum 1 Januari 2020
                logger.warning(f"Timestamp too old ({normalized_timestamp}), skipping document {doc.id}")
                continue
            elif normalized_timestamp > current_time + 86400:  # Lebih dari 1 hari ke depan
                logger.warning(f"Timestamp too far in future ({normalized_timestamp}), skipping document {doc.id}")
                continue
            
            # Filter berdasarkan timestamp jika parameter diberikan
            if start_param and end_param:
                if normalized_timestamp < start_param or normalized_timestamp > end_param:
                    continue

            firestore_data.append({
                'timestamp': normalized_timestamp,
                'humidity': float(humidity_firestore) if humidity_firestore else 0,
                'temperature': float(temperature_firestore) if temperature_firestore else 0,
                'soil_moisture': float(soil_moisture_firestore) if soil_moisture_firestore else 0,
                'source': 'firestore',
                'original_datetime': datetime_firestore,
                'original_timestamp': timestamp_firestore,
                'doc_id': doc.id
            })

        # Jika tidak ada parameter waktu atau data Firestore kosong, tambahkan data dari Realtime Database
        if not start_param or not end_param or len(firestore_data) == 0:
            # Ambil data dari Firebase Realtime Database
            ref_dht = db.reference('/DHT')
            dht_data = ref_dht.get()

            ref_soil = db.reference('/SoilMoisture')
            soil_moisture_data = ref_soil.get()

            # Data Realtime Database (data terbaru)
            if dht_data and soil_moisture_data:
                humidity = dht_data.get('humidity', 0)
                temperature = dht_data.get('temperature', 0)
                soil_percentage = soil_moisture_data.get('percentage', 0)
                latest_update = soil_moisture_data.get('latestUpdate', '')

                # Normalisasi timestamp
                if latest_update:
                    normalized_timestamp = parse_datetime_string(latest_update)
                    if not normalized_timestamp:
                        normalized_timestamp = normalize_timestamp(latest_update)
                else:
                    normalized_timestamp = datetime.now().timestamp()
                
                # Cek apakah data realtime database masuk dalam rentang waktu yang diminta
                include_realtime = True
                if start_param and end_param:
                    if normalized_timestamp < start_param or normalized_timestamp > end_param:
                        include_realtime = False
                
                if include_realtime:
                    firestore_data.append({
                        'timestamp': normalized_timestamp,
                        'humidity': float(humidity) if humidity else 0,
                        'temperature': float(temperature) if temperature else 0,
                        'soil_moisture': float(soil_percentage) if soil_percentage else 0,
                        'source': 'realtime_db'
                    })

        # Sorting data berdasarkan metode yang dipilih
        if sort_method == 'datetime':
            logger.info("Sorting data by detailed datetime (year -> month -> day -> hour -> minute -> second)")
            firestore_data.sort(key=get_detailed_sort_key)
        else:
            logger.info("Sorting data by timestamp")
            firestore_data.sort(key=lambda x: x['timestamp'])
        
        # Batasi jumlah data sesuai limit
        if len(firestore_data) > limit_param:
            firestore_data = firestore_data[-limit_param:]
            # Urutkan ulang data yang sudah dibatasi dengan metode yang sama
            if sort_method == 'datetime':
                firestore_data.sort(key=get_detailed_sort_key)
            else:
                firestore_data.sort(key=lambda x: x['timestamp'])

        logger.info(f"Returning {len(firestore_data)} data points (sorted by {sort_method})")
        
        if firestore_data:
            first_dt = datetime.fromtimestamp(firestore_data[0]['timestamp'])
            last_dt = datetime.fromtimestamp(firestore_data[-1]['timestamp'])
            logger.info(f"Time range: {first_dt.strftime('%Y-%m-%d %H:%M:%S')} to {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Debug: tampilkan beberapa timestamp untuk verifikasi urutan
            if len(firestore_data) > 5:
                logger.info("Sample timestamps (first 5 records):")
                for i in range(min(5, len(firestore_data))):
                    dt = datetime.fromtimestamp(firestore_data[i]['timestamp'])
                    original_dt = firestore_data[i].get('original_datetime', 'N/A')
                    original_ts = firestore_data[i].get('original_timestamp', 'N/A')
                    logger.info(f"  {i+1}. Parsed: {dt.strftime('%Y-%m-%d %H:%M:%S')}, Original DateTime: {original_dt}, Original Timestamp: {original_ts}, Source: {firestore_data[i].get('source', 'unknown')}")

        # Bersihkan data sebelum mengirim (hapus field debug)
        clean_data = []
        for item in firestore_data:
            clean_item = {
                'timestamp': item['timestamp'],
                'humidity': item['humidity'],
                'temperature': item['temperature'],
                'soil_moisture': item['soil_moisture'],
                'source': item['source']
            }
            clean_data.append(clean_item)

        return jsonify(clean_data)

    except Exception as e:
        logger.error(f"Error in get_sensor_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-data')
def get_latest_data():
    """Endpoint untuk mendapatkan data sensor terbaru dengan timestamp yang benar."""
    try:
        # Ambil data terbaru dari Firebase Realtime Database
        ref_dht = db.reference('/DHT')
        dht_data = ref_dht.get()
        
        ref_soil = db.reference('/SoilMoisture')
        soil_data = ref_soil.get()
        
        print("DHT Data:", dht_data)
        print("Soil Data:", soil_data)
        
        # Inisialisasi nilai default
        suhu_udara = 0
        kelembapan_udara = 0
        kelembapan_tanah = 0
        timestamp = datetime.now().timestamp()
        
        # Ambil data DHT
        if dht_data:
            if isinstance(dht_data, dict):
                suhu_udara = dht_data.get('temperature', 0)
                kelembapan_udara = dht_data.get('humidity', 0)
                # Jika ada latestUpdate, parse dengan fungsi yang diperbaiki
                if 'latestUpdate' in dht_data:
                    parsed_timestamp = parse_datetime_string(dht_data.get('latestUpdate'))
                    if parsed_timestamp:
                        timestamp = parsed_timestamp
                    else:
                        timestamp = normalize_timestamp(dht_data.get('latestUpdate'))
            else:
                print(f"DHT data is not dict: {type(dht_data)}")
        
        # Ambil data kelembapan tanah
        if soil_data:
            if isinstance(soil_data, dict):
                kelembapan_tanah = soil_data.get('percentage', 0)
                # Update timestamp jika soil data memiliki latestUpdate yang lebih baru
                if 'latestUpdate' in soil_data:
                    parsed_timestamp = parse_datetime_string(soil_data.get('latestUpdate'))
                    if parsed_timestamp:
                        soil_timestamp = parsed_timestamp
                    else:
                        soil_timestamp = normalize_timestamp(soil_data.get('latestUpdate'))
                    
                    # Gunakan timestamp yang lebih baru
                    if soil_timestamp > timestamp:
                        timestamp = soil_timestamp
            else:
                print(f"Soil data is not dict: {type(soil_data)}")
        
        result = {
            'suhu_udara': float(suhu_udara) if suhu_udara else 0,
            'kelembapan_udara': float(kelembapan_udara) if kelembapan_udara else 0,
            'kelembapan_tanah': float(kelembapan_tanah) if kelembapan_tanah else 0,
            'timestamp': timestamp
        }
        
        print("Returning result:", result)
        return jsonify(result)

    except Exception as e:
        print("Error in get_latest_data:", str(e))
        return jsonify({
            'error': str(e),
            'suhu_udara': 0,
            'kelembapan_udara': 0,
            'kelembapan_tanah': 0,
            'timestamp': datetime.now().timestamp()
        }), 500

# Endpoint untuk debugging timestamp
@app.route('/api/debug-timestamps')
def debug_timestamps():
    """Endpoint untuk debugging timestamp di Firestore dengan informasi lebih detail."""
    try:
        collection_ref = firestore_client.collection('sensor_history')
        docs = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        
        debug_data = []
        for doc in docs:
            data = doc.to_dict()
            
            timestamp_field = data.get('timestamp')
            datetime_field = data.get('DateTime')
            
            # Test parsing dengan fungsi yang diperbaiki
            parsed_datetime = None
            if datetime_field:
                parsed_datetime = parse_datetime_string(datetime_field)
            
            normalized_ts = None
            if timestamp_field:
                normalized_ts = normalize_timestamp(timestamp_field)
            
            debug_info = {
                'doc_id': doc.id,
                'timestamp_field': timestamp_field,
                'timestamp_type': type(timestamp_field).__name__,
                'datetime_field': datetime_field,
                'datetime_type': type(datetime_field).__name__,
                'parsed_datetime_timestamp': parsed_datetime,
                'normalized_timestamp': normalized_ts,
                'final_timestamp': parsed_datetime if parsed_datetime else normalized_ts,
                'formatted_datetime': datetime.fromtimestamp(parsed_datetime if parsed_datetime else (normalized_ts if normalized_ts else datetime.now().timestamp())).strftime('%Y-%m-%d %H:%M:%S')
            }
            debug_data.append(debug_info)
        
        return jsonify({
            'success': True,
            'debug_data': debug_data,
            'count': len(debug_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Sisanya sama seperti kode asli...
def scheduled_watering_analysis():
    """
    Fungsi untuk melakukan analisis penyiraman terjadwal.
    Dipanggil otomatis oleh scheduler.
    """
    try:
        logger.info("Memulai analisis penyiraman terjadwal...")
        
        # Ambil data terbaru dari sensor_history
        collection_ref = firestore_client.collection('sensor_history')
        docs = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
        
        latest_data = None
        for doc in docs:
            latest_data = doc.to_dict()
            break
            
        if not latest_data:
            logger.error("Tidak ada data sensor tersedia untuk analisis terjadwal")
            return
            
        # Ekstrak data sensor
        temperature = float(latest_data.get('temperature', 0))
        humidity = float(latest_data.get('humidity', 0))
        soil_moisture = float(latest_data.get('soil_moisture', 0))
        
        # Lakukan prediksi berdasarkan ambang batas
        watering_decision = check_watering_conditions(temperature, humidity, soil_moisture)
        
        # Tentukan alasan keputusan
        reasons = []
        if temperature < 20:
            reasons.append(f"Suhu terlalu rendah ({temperature}°C < 20°C)")
        elif temperature > 31:
            reasons.append(f"Suhu terlalu tinggi ({temperature}°C > 31°C)")
            
        if humidity < 75:
            reasons.append(f"Kelembapan udara terlalu rendah ({humidity}% < 75%)")
        elif humidity > 80:
            reasons.append(f"Kelembapan udara terlalu tinggi ({humidity}% > 80%)")
            
        if soil_moisture < 60:
            reasons.append(f"Kelembapan tanah terlalu rendah ({soil_moisture}% < 60%)")
        elif soil_moisture > 75:
            reasons.append(f"Kelembapan tanah terlalu tinggi ({soil_moisture}% > 75%)")
        
        decision_text = "siram" if watering_decision == 1 else "jangan siram"
        reason_text = "; ".join(reasons) if reasons else "Semua parameter dalam rentang optimal"
        
        # Waktu saat ini
        current_time = datetime.now()
        
        # Data untuk disimpan ke watering_analysis
        analysis_result = {
            'humidity': humidity,
            'soil_moisture': soil_moisture, 
            'temperature': temperature,
            'time': current_time.isoformat(),
            'timestamp': current_time.timestamp(),
            'keputusan_penyiraman': watering_decision,
            'keputusan_text': decision_text,
            'alasan': reason_text,
            'created_at': current_time.isoformat(),
            'analysis_type': 'scheduled',
            'scheduled_time': current_time.strftime('%H:%M')
        }
        
        # Jika model Random Forest tersedia, tambahkan prediksi RF
        if rf_model is not None:
            try:
                features = np.array([[temperature, humidity, soil_moisture]])
                rf_prediction = rf_model.predict(features)[0]
                rf_probability = rf_model.predict_proba(features)[0]
                
                analysis_result.update({
                    'rf_prediction': int(rf_prediction),
                    'rf_confidence': float(max(rf_probability) * 100),
                    'rf_probability_no_water': float(rf_probability[0]),
                    'rf_probability_water': float(rf_probability[1])
                })
            except Exception as e:
                logger.error(f"Error using Random Forest model in scheduled analysis: {e}")
        
        # Simpan hasil analisis ke koleksi 'watering_analysis'
        analysis_collection = firestore_client.collection('watering_analysis')
        doc_ref = analysis_collection.add(analysis_result)
        
        logger.info(f"Analisis terjadwal berhasil disimpan dengan ID: {doc_ref[1].id}")
        logger.info(f"Keputusan: {decision_text} - Alasan: {reason_text}")
        logger.info(f"Data sensor - Suhu: {temperature}°C, Kelembapan Udara: {humidity}%, Kelembapan Tanah: {soil_moisture}%")
        
    except Exception as e:
        logger.error(f"Error in scheduled watering analysis: {str(e)}")

# Inisialisasi scheduler
scheduler = BackgroundScheduler()

# Tambahkan job terjadwal untuk analisis penyiraman
scheduled_hours = [6, 8, 10, 12, 14, 16, 18]

for hour in scheduled_hours:
    scheduler.add_job(
        func=scheduled_watering_analysis,
        trigger=CronTrigger(hour=hour, minute=0),
        id=f'watering_analysis_{hour:02d}00',
        name=f'Analisis Penyiraman {hour:02d}:00',
        replace_existing=True
    )

# Mulai scheduler
scheduler.start()
logger.info("Scheduler untuk analisis penyiraman otomatis telah dimulai")
logger.info(f"Jadwal analisis: {', '.join([f'{h:02d}:00' for h in scheduled_hours])}")

# Shutdown scheduler saat aplikasi ditutup
atexit.register(lambda: scheduler.shutdown())

@app.route('/')
def index():
    """Halaman utama yang menampilkan grafik data sensor."""
    return render_template('index.html')

@app.route('/api/analyze-watering')
def analyze_watering():
    """
    Endpoint untuk melakukan analisis penyiraman berdasarkan ambang batas yang ditentukan.
    Data diambil dari koleksi sensor_history dan hasil disimpan ke watering_analysis.
    """
    try:
        # Ambil data terbaru dari sensor_history
        collection_ref = firestore_client.collection('sensor_history')
        docs = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
        
        latest_data = None
        for doc in docs:
            latest_data = doc.to_dict()
            break
            
        if not latest_data:
            return jsonify({'error': 'Tidak ada data sensor tersedia di sensor_history'}), 404
            
        # Ekstrak data sensor
        temperature = float(latest_data.get('temperature', 0))
        humidity = float(latest_data.get('humidity', 0))
        soil_moisture = float(latest_data.get('soil_moisture', 0))
        
        # Lakukan prediksi berdasarkan ambang batas
        watering_decision = check_watering_conditions(temperature, humidity, soil_moisture)
        
        # Tentukan alasan keputusan
        reasons = []
        if temperature < 20:
            reasons.append(f"Suhu terlalu rendah ({temperature}°C < 20°C)")
        elif temperature > 31:
            reasons.append(f"Suhu terlalu tinggi ({temperature}°C > 31°C)")
            
        if humidity < 75:
            reasons.append(f"Kelembapan udara terlalu rendah ({humidity}% < 75%)")
        elif humidity > 80:
            reasons.append(f"Kelembapan udara terlalu tinggi ({humidity}% > 80%)")
            
        if soil_moisture < 60:
            reasons.append(f"Kelembapan tanah terlalu rendah ({soil_moisture}% < 60%)")
        elif soil_moisture > 75:
            reasons.append(f"Kelembapan tanah terlalu tinggi ({soil_moisture}% > 75%)")
        
        decision_text = "siram" if watering_decision == 1 else "jangan siram"
        reason_text = "; ".join(reasons) if reasons else "Semua parameter dalam rentang optimal"
        
        # Waktu saat ini
        current_time = datetime.now()
        
        # Data untuk disimpan ke watering_analysis sesuai permintaan
        analysis_result = {
            'humidity': humidity,
            'soil_moisture': soil_moisture, 
            'temperature': temperature,
            'time': current_time.isoformat(),
            'timestamp': current_time.timestamp(),
            'keputusan_penyiraman': watering_decision,
            'keputusan_text': decision_text,
            'alasan': reason_text,
            'created_at': current_time.isoformat(),
            'analysis_type': 'manual'
        }
        
        # Jika model Random Forest tersedia, tambahkan prediksi RF sebagai perbandingan
        if rf_model is not None:
            try:
                features = np.array([[temperature, humidity, soil_moisture]])
                rf_prediction = rf_model.predict(features)[0]
                rf_probability = rf_model.predict_proba(features)[0]
                
                analysis_result.update({
                    'rf_prediction': int(rf_prediction),
                    'rf_confidence': float(max(rf_probability) * 100),
                    'rf_probability_no_water': float(rf_probability[0]),
                    'rf_probability_water': float(rf_probability[1])
                })
            except Exception as e:
                print(f"Error using Random Forest model: {e}")
        
        # Simpan hasil analisis ke koleksi 'watering_analysis'
        analysis_collection = firestore_client.collection('watering_analysis')
        doc_ref = analysis_collection.add(analysis_result)
        
        print(f"Analisis berhasil disimpan dengan ID: {doc_ref[1].id}")
        print(f"Keputusan: {decision_text} - Alasan: {reason_text}")
        
        # Return hasil analisis
        response_data = {
            'success': True,
            'prediction': watering_decision,
            'decision': decision_text,
            'reason': reason_text,
            'sensor_data': {
                'temperature': temperature,
                'humidity': humidity,
                'soil_moisture': soil_moisture
            },
            'thresholds': {
                'temperature_range': '20-31°C',
                'humidity_range': '75-80%',
                'soil_moisture_range': '60-75%'
            },
            'timestamp': analysis_result['timestamp'],
            'time': analysis_result['time'],
            'doc_id': doc_ref[1].id
        }
        
        # Tambahkan data Random Forest jika tersedia
        if 'rf_prediction' in analysis_result:
            response_data['random_forest'] = {
                'prediction': analysis_result['rf_prediction'],
                'confidence': analysis_result['rf_confidence'],
                'probabilities': {
                    'no_water': analysis_result['rf_probability_no_water'],
                    'water': analysis_result['rf_probability_water']
                }
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_watering: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watering-history')
def get_watering_history():
    """Endpoint untuk mendapatkan riwayat analisis penyiraman."""
    try:
        # Ambil riwayat analisis dari Firestore, diurutkan berdasarkan timestamp terbaru
        analysis_collection = firestore_client.collection('watering_analysis')
        docs = analysis_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20).stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            history.append(data)
            
        return jsonify({
            'success': True,
            'data': history,
            'count': len(history)
        })
        
    except Exception as e:
        print(f"Error in get_watering_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threshold-info')
def get_threshold_info():
    """Endpoint untuk mendapatkan informasi ambang batas yang digunakan."""
    return jsonify({
        'thresholds': {
            'temperature': {
                'min': 20,
                'max': 31,
                'unit': '°C',
                'description': 'Rentang suhu udara optimal'
            },
            'humidity': {
                'min': 75,
                'max': 80,
                'unit': '%',
                'description': 'Rentang kelembapan udara optimal'
            },
            'soil_moisture': {
                'min': 60,
                'max': 75,
                'unit': '%',
                'description': 'Rentang kelembapan tanah optimal'
            }
        },
        'logic': 'Jika salah satu parameter di luar rentang optimal, maka keputusan = siram (1), jika tidak = jangan siram (0)'
    })

@app.route('/api/scheduler-status')
def get_scheduler_status():
    """Endpoint untuk mendapatkan status scheduler dan jadwal analisis."""
    try:
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return jsonify({
            'scheduler_running': scheduler.running,
            'scheduled_hours': scheduled_hours,
            'jobs': jobs,
            'total_jobs': len(jobs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger-scheduled-analysis')
def trigger_scheduled_analysis():
    """Endpoint untuk memicu analisis terjadwal secara manual (untuk testing)."""
    try:
        scheduled_watering_analysis()
        return jsonify({
            'success': True,
            'message': 'Analisis terjadwal berhasil dijalankan secara manual'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
