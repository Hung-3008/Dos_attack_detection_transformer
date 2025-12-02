from flask import Flask, render_template, request, Response, jsonify
import time
import json
import threading
from queue import Queue

app = Flask(__name__)

# Queue to store messages for SSE
msg_queue = Queue()

# Buffer for the detection system
packet_buffer = []
packet_buffer_lock = threading.Lock()

# Blacklist for blocked IPs
BLACKLIST = set()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log', methods=['POST'])
def log_packet():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    src_ip = data.get('src_ip')
    
    # Check Blacklist
    if src_ip in BLACKLIST:
        return jsonify({"status": "blocked", "message": "IP is blacklisted"}), 403
    
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = time.strftime("%H:%M:%S")
        
    # Put data into the SSE queue
    msg_queue.put(data)
    
    # Add to detection buffer
    with packet_buffer_lock:
        packet_buffer.append(data)
        
    return jsonify({"status": "success"}), 200

@app.route('/fetch_packets', methods=['GET'])
def fetch_packets():
    """Endpoint for the Real-time Detector to fetch recent packets."""
    global packet_buffer
    with packet_buffer_lock:
        # Return copy and clear buffer
        data = list(packet_buffer)
        packet_buffer.clear()
    return jsonify(data), 200

@app.route('/block_ip', methods=['POST'])
def block_ip():
    """Endpoint for the Mitigator to block an IP."""
    data = request.json
    ip_to_block = data.get('ip')
    if ip_to_block:
        BLACKLIST.add(ip_to_block)
        print(f"!!! BLOCKED IP: {ip_to_block} !!!")
        # Notify dashboard via SSE (special alert type)
        alert = {
            "src_ip": ip_to_block,
            "message": "BLOCKED BY FIREWALL",
            "type": "dos",
            "timestamp": time.strftime("%H:%M:%S")
        }
        msg_queue.put(alert)
        return jsonify({"status": "success", "blocked": ip_to_block}), 200
    return jsonify({"status": "error", "message": "No IP provided"}), 400

def event_stream():
    while True:
        if not msg_queue.empty():
            data = msg_queue.get()
            # SSE format: data: <payload>\n\n
            yield f"data: {json.dumps(data)}\n\n"
        else:
            # Send a keep-alive comment to keep connection open
            # yield ": keep-alive\n\n"
            time.sleep(0.1)

@app.route('/stream')
def stream():
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Run threaded=True to handle multiple requests (SSE + POSTs)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
