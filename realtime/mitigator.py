import requests

class Mitigator:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.blocked_ips = set()

    def block_ip(self, ip):
        if ip in self.blocked_ips:
            return
        
        try:
            print(f"[Mitigator] Blocking IP: {ip}")
            response = requests.post(f"{self.server_url}/block_ip", json={"ip": ip}, timeout=2)
            if response.status_code == 200:
                self.blocked_ips.add(ip)
                print(f"[Mitigator] Successfully blocked {ip}")
            else:
                print(f"[Mitigator] Failed to block {ip}: {response.text}")
        except Exception as e:
            print(f"[Mitigator] Error contacting server: {e}")
