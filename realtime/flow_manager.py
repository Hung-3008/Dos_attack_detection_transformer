import requests
import pandas as pd
import numpy as np
import time
from collections import defaultdict

class FlowManager:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        # Default values for features we can't easily measure from HTTP logs
        self.defaults = {
            'proto': 'tcp',
            'service': 'http',
            'state': 'FIN',
            'dbytes': 268, # Average response size
            'dpkts': 2,    # Average response packets
            'sttl': 254,
            'dttl': 252,
            'sloss': 0,
            'dloss': 0,
            'sjit': 30.0,
            'djit': 30.0,
            'swin': 255,
            'stcpb': 100000,
            'dtcpb': 100000,
            'dwin': 255,
            'tcprtt': 0.0,
            'synack': 0.0,
            'ackdat': 0.0,
            'smean': 57,
            'dmean': 89,
            'trans_depth': 0,
            'response_body_len': 100,
            'ct_srv_src': 1,
            'ct_state_ttl': 0,
            'ct_dst_ltm': 1,
            'ct_src_dport_ltm': 1,
            'ct_dst_sport_ltm': 1,
            'ct_dst_src_ltm': 1,
            'is_ftp_login': 0,
            'ct_ftp_cmd': 0,
            'ct_flw_http_mthd': 1,
            'ct_src_ltm': 1,
            'ct_srv_dst': 1,
            'is_sm_ips_ports': 0
        }

    def fetch_packets(self):
        try:
            resp = requests.get(f"{self.server_url}/fetch_packets", timeout=1)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def process_packets(self, packets):
        """
        Convert raw packet logs into feature vectors per source IP.
        """
        if not packets:
            return pd.DataFrame()

        # Group by Source IP
        groups = defaultdict(list)
        for p in packets:
            groups[p['src_ip']].append(p)

        features_list = []
        
        for src_ip, pkts in groups.items():
            # Calculate features for this batch
            count = len(pkts)
            
            # Approximate duration: max 1 second for the batch or based on timestamps if available
            # For simplicity in this demo, we assume the batch represents a 1-second window
            dur = 1.0 
            
            # sbytes: sum of message lengths + HTTP headers (approx 300 bytes)
            sbytes = sum(len(p.get('message', '')) + 300 for p in pkts)
            
            # spkts: count
            spkts = count
            
            # sload: bits per second
            sload = (sbytes * 8) / dur
            
            # sinpkt: average inter-arrival time
            sinpkt = (dur * 1000) / count if count > 0 else 0
            
            # Create feature dict
            row = self.defaults.copy()
            row['src_ip'] = src_ip # Keep track of IP
            row['dur'] = dur
            row['sbytes'] = sbytes
            row['spkts'] = spkts
            row['sload'] = sload
            row['sinpkt'] = sinpkt
            
            # Update dynamic counts (simple approximation)
            row['ct_srv_src'] = min(count, 100)
            row['ct_dst_ltm'] = min(count, 100)
            row['ct_src_ltm'] = min(count, 100)
            row['ct_srv_dst'] = min(count, 100)
            
            # Debug print for high traffic
            if spkts > 10:
                print(f"[DEBUG] {src_ip}: spkts={spkts}, sbytes={sbytes}, sload={sload:.2f}, sinpkt={sinpkt:.2f}")

            features_list.append(row)

        return pd.DataFrame(features_list)
