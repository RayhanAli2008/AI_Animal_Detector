import io
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

# 1. Load your model once when the API starts up
model_path = r"C:\Users\Rayhan\Downloads\AnimalGuessr\animalGuesser.keras"
model = tf.keras.models.load_model(model_path)

class_names = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 
    'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 
    'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 
    'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 
    'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 
    'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 
    'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 
    'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 
    'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 
    'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 
    'wombat', 'woodpecker', 'zebra'
]

@app.post("/predict")
async def predict_animal(file: UploadFile = File(...)):
    # 2. Read the uploaded image bytes and convert to an OpenCV image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image file format."}

    # 3. Preprocess the image exactly like your original script
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = cv2.resize(processed_frame, (224, 224))
    
    img_array = tf.expand_dims(processed_frame, 0)
    img_array = tf.cast(img_array, tf.float32)

    # 4. Run the prediction
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    max_idx = np.argmax(score)
    confidence = score[max_idx] # This is a float between 0.0 and 1.0

    # 5. Check if confidence is 70% (0.70) or higher
    # 5. Check if confidence is 70% (0.70) or higher
    if confidence >= 0.70:
        winning_animal = class_names[max_idx]
        result_text = f"{winning_animal} ({confidence * 100:.2f}%)"
        
        h, w, _ = frame.shape
        
        # Dynamically scale font size based on the image width
        font_scale = max(0.5, min(w / 500.0, 1.0))
        thickness = max(1, int(font_scale * 2))
        
        # Calculate exactly how much space the text takes up
        (text_w, text_h), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 1. Draw the outer frame box
        cv2.rectangle(frame, (15, 15), (w - 15, h - 15), (0, 255, 0), 3)
        
        # 2. Draw a solid green banner for the text background in the top-left corner
        cv2.rectangle(frame, (15, 15), (15 + text_w + 20, 15 + text_h + 20), (0, 255, 0), -1)
        
        # 3. Draw the text in clean black ink over the green banner
        cv2.putText(frame, result_text, (25, 15 + text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    else:
        # If confidence is less than 70%, don't alter the image
        pass

    # 6. Convert the OpenCV image back into bytes to send over the API
    _, encoded_img = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)