import requests
import time
from PIL import Image
import numpy as np
import io

# Create a simple test image
img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# Send requests in a loop
for i in range(20):
    try:
        response = requests.post(
            'http://localhost:8000/predict',
            files={'file': ('test.png', img_byte_arr, 'image/png')}
        )
        print(f"Request {i+1}: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error on request {i+1}: {str(e)}")
    
    # Random delay between 0.5 and 2 seconds
    time.sleep(np.random.uniform(0.5, 2.0))
