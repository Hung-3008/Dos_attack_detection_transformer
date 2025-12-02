import argparse
import time
import threading
import random
import requests
import sys

# Configuration
TARGET_URL = "http://localhost:5000/log"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/89.0"
]

# Fixed pool of IPs for simulation to allow per-IP detection
IP_POOL = [f"192.168.1.{i}" for i in range(100, 120)] # 20 Botnet IPs

def generate_packet(attack_type="normal"):
    """Generate a random packet payload."""
    
    if attack_type == "normal":
        # Normal users have their own IPs (simulated)
        src_ip = f"10.0.0.{random.randint(1, 50)}"
        messages = ["Hello server", "Just checking in", "Status update", "Data sync", "Ping"]
        msg = random.choice(messages)
    else:
        # DoS attackers come from the botnet pool
        src_ip = random.choice(IP_POOL)
        # DoS payload: random junk or repetitive data
        msg = "FLOOD_" + "".join(random.choices("ABCDEF0123456789", k=10))
        
    return {
        "src_ip": src_ip,
        "message": msg,
        "type": attack_type,
        "user_agent": random.choice(USER_AGENTS)
    }

def send_packet(session, attack_type="normal"):
    """Send a single packet to the server."""
    try:
        data = generate_packet(attack_type)
        resp = session.post(TARGET_URL, json=data, timeout=1)
        # print(f"[{resp.status_code}] Sent {attack_type} packet")
    except Exception as e:
        # In DoS mode, errors are expected (connection refused, timeout)
        if attack_type == "normal":
            print(f"Error sending packet: {e}")
        pass

def normal_traffic():
    """Simulate normal user traffic."""
    print(f"[*] Starting NORMAL traffic simulation to {TARGET_URL}")
    session = requests.Session()
    try:
        while True:
            send_packet(session, "normal")
            # Random delay between 0.5s and 2.0s
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print("\n[*] Stopped normal traffic.")

def dos_attack_thread():
    """Single thread for DoS attack."""
    session = requests.Session()
    while True:
        send_packet(session, "dos")

def dos_attack(threads=50):
    """Simulate DoS attack with multiple threads."""
    print(f"[*] Starting DoS attack simulation to {TARGET_URL}")
    print(f"[*] Launching {threads} threads...")
    
    thread_list = []
    for _ in range(threads):
        t = threading.Thread(target=dos_attack_thread)
        t.daemon = True  # Daemon threads exit when main program exits
        t.start()
        thread_list.append(t)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopped DoS attack.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Traffic Simulator")
    parser.add_argument("--mode", choices=["normal", "dos"], required=True, help="Traffic mode: normal or dos")
    parser.add_argument("--threads", type=int, default=50, help="Number of threads for DoS mode")
    
    args = parser.parse_args()
    
    # Check if server is up
    try:
        requests.get("http://localhost:5000", timeout=2)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server at http://localhost:5000")
        print("Please make sure the Flask server is running first.")
        sys.exit(1)

    if args.mode == "normal":
        normal_traffic()
    else:
        dos_attack(args.threads)
