python main.py --model transformer --device cuda --epochs 5

ls -lah checkpoints

python demo/gradio_demo.py

# start server
python server/app.py 
python realtime/main.py
python attack/attack_http.py --mode normal
python attack/attack_http.py --mode dos