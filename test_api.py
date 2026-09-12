import requests
import numpy as np

URL = "http://127.0.0.1:5000/predict"

# 10 frames, 64x64 resolution, 3 color channels
sample_sequence = np.random.rand(10, 64, 64, 3).tolist()

payload = {
    "sequence": sample_sequence
}

print("Sending test sequence to Flask server...")
response = requests.post(URL, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())