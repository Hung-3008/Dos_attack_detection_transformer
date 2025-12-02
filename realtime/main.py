import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.flow_manager import FlowManager
from realtime.detector import Detector
from realtime.mitigator import Mitigator

def main():
    print("[*] Starting Real-time DoS Prevention System (User-Space)...")
    
    # Paths
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    # Find the latest checkpoint
    # Hardcoding the one we found earlier for reliability
    model_path = os.path.join(ckpt_dir, "transformer_20251102_210010.pth")
    meta_path = os.path.join(ckpt_dir, "transformer_20251102_210010_meta.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Initialize components
    flow_manager = FlowManager()
    mitigator = Mitigator()
    detector = Detector(model_path, meta_path)
    
    print("[*] System Ready. Monitoring traffic...")
    
    try:
        while True:
            # 1. Fetch and Process Packets
            packets = flow_manager.fetch_packets()
            if packets:
                feature_df = flow_manager.process_packets(packets)
                
                if not feature_df.empty:
                    # --- HEURISTIC CHECK (FAST PATH) ---
                    # If an IP is sending too many packets, block immediately
                    # This covers high-rate floods that might confuse the model or cause latency
                    RATE_LIMIT = 50 # packets per fetch (approx 1 sec)
                    
                    for i, row in feature_df.iterrows():
                        src_ip = row['src_ip']
                        spkts = row['spkts']
                        
                        if spkts > RATE_LIMIT:
                            print(f"[ALERT] High Rate Detected from {src_ip} ({spkts} pkts/sec). Blocking immediately.")
                            mitigator.block_ip(src_ip)
                            # Drop this row from model inference to save resources
                            feature_df = feature_df.drop(i)
                    
                    # --- MODEL INFERENCE (SLOW PATH) ---
                    if not feature_df.empty:
                        results = detector.predict(feature_df)
                        
                        for i, (label, conf) in enumerate(results):
                            # Re-fetch IP because we might have dropped rows
                            # Use iloc to safely access by position in the remaining dataframe
                            src_ip = feature_df.iloc[i]['src_ip']
                            
                            # Label 1 is Attack (DoS)
                            if label == 1:
                                print(f"[ALERT] Model Detected DoS from {src_ip} (Conf: {conf:.2f})")
                                mitigator.block_ip(src_ip)
                            else:
                                # Optional: Print normal traffic stats for debugging
                                # print(f"[INFO] Normal: {src_ip} (Conf: {conf:.2f})")
                                pass
            
            # Sleep briefly to accumulate packets
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping system.")

if __name__ == "__main__":
    main()
