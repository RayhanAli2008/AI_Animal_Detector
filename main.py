import numpy as np
import tensorflow as tf
import cv2
print('hi')
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

model_path  = r"C:\Users\Rayhan\Downloads\animalGuesser.keras"
model       = tf.keras.models.load_model(model_path)

#Load the webcam
camera      = cv2.VideoCapture(0)

while True:
    current_web_status = camera.read()
    boolean_status     = current_web_status[0]
    frame              = current_web_status[1]

    if boolean_status == False:
        print("unable to get frame")
        break

    #process the frame, change it from bgr to rgb
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #resize
    processed_frame = cv2.resize(processed_frame, (224, 224))
    #convert to expected input for tensorflow, the image normally has 3 dimensions (height,width,color_channels), but for tensorflow they need a "batch" parameter wo we insert at index 0 a batch parameter and the dafult val is 1 meaning a batch of one image
    img_array = tf.expand_dims(processed_frame, 0)