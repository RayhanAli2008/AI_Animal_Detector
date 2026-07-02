import numpy as np
import tensorflow as tf
import cv2

print('hi')
#all possible animals
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

model_path  = r"C:\Users\Rayhan\Downloads\animalGuesser.keras"
model       = tf.keras.models.load_model(model_path)
camera      = cv2.VideoCapture(0)

while True:
    current_web_status = camera.read()
    boolean_status     = current_web_status[0]
    frame              = current_web_status[1]

    if boolean_status == False:
        print("unable to get frame")
        break

    # Process the frame, change it from bgr to rgb
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    processed_frame = cv2.resize(processed_frame, (224, 224))
    
    # Convert to expected input for tensorflow by adding a batch parameter
    img_array = tf.expand_dims(processed_frame, 0)
    img_array = tf.cast(img_array, tf.float32) 

    # Make the prediction (verbose=0 stops TensorFlow from flooding your console with progress bars)
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]

    
    winning_animal = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    result_text = f"{winning_animal} ({confidence:.2f}%)"

    # Draw the results directly onto the live camera frame
    # (frame, text, coordinates, font, font scale, color in BGR, thickness)
    cv2.putText(frame, result_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow("Animal Guesser - Live Feed", frame)

    # Press 'q' on your keyboard to quit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
print("Application closed successfully.")