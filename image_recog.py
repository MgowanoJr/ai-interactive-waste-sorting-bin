# Example code provided by TensorFlow Authors and @jiteshsaini have been useful in creating this project.
# Author: Andre Prosper
# Project: Interactive Waste Sorting Bin


from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
from time import sleep
import cv2
import os
import RPi.GPIO as GPIO
import torch


GPIO.setwarnings(False)

cap = cv2.VideoCapture(0)

font=cv2.FONT_HERSHEY_SIMPLEX
text_overlay=""
prev_lbl="None"
lbl="None"

plastic_green=22
paper_green=17
food_green=23
metal_green=24



#---------Set up GPIO Pins-----------------------------

GPIO.setmode(GPIO.BCM)
#GPIO.setup(plastic_red,GPIO.OUT)
GPIO.setup(plastic_green,GPIO.OUT)

#GPIO.setup(paper_red,GPIO.OUT)
GPIO.setup(paper_green,GPIO.OUT)


#GPIO.setup(food_red,GPIO.OUT)
GPIO.setup(food_green,GPIO.OUT)

GPIO.setup(metal_green,GPIO.OUT)

import sys
sys.path.insert(0, '/var/www/iwsb')

#------------------------------------------------------
       
#---------Flask----------------------------------------
from flask import Flask, Response
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return "Default Message"
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
#-------------------------------------------------------------


#-----initialise the Model and Load into interpreter-------------------------

#specify the path of Model and Label file
model_path = "/var/www/iwsb/all_models/model_unquant.tflite" 
label_path = "/var/www/iwsb/all_models/labels_unquant.txt"
top_k_results = 3
with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Load TFLite model and allocate tensors

interpreter = Interpreter(model_path=model_path)
print(interpreter)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

threshold=0.75 #accuracy to be met for light indication
#-----------------------------------------------------------

#Control GPIOs and give feedback 
def action(pred,lbl):
        global prev_lbl
        global text_overlay
        

        if (pred < threshold and prev_lbl != lbl):
                text_overlay = "__"
                GPIO.output(plastic_green,False)
                GPIO.output(paper_green,False)
                GPIO.output(food_green,False)
                GPIO.output(metal_green,False)

        if (pred >= threshold):

                percent=round(pred*100)

                if lbl == "Plastic":
                    GPIO.output(plastic_green,True)
                    GPIO.output(paper_green,False)
                    GPIO.output(food_green,False)
                    GPIO.output(metal_green,False)

                if lbl == "Paper":
                    GPIO.output(paper_green,True)
                    GPIO.output(plastic_green,False)
                    GPIO.output(food_green,False)
                    GPIO.output(metal_green,False)

                if lbl == "Food waste":
                    GPIO.output(food_green,True)
                    GPIO.output(plastic_green,False)
                    GPIO.output(paper_green,False)
                    GPIO.output(metal_green,False)

                if lbl == "Metal":
                    GPIO.output(metal_green,True)
                    GPIO.output(plastic_green,False)
                    GPIO.output(paper_green,False)
                    GPIO.output(food_green,False)

                if lbl == "None":
                    GPIO.output(plastic_green,False)
                    GPIO.output(paper_green,False)
                    GPIO.output(food_green,False)
                    GPIO.output(metal_green,False)

                text_overlay= "Saw a " + lbl + ", i am " + str(percent) + "% sure"
                prev_lbl=lbl

                #sleep(1)

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def main():
        while True:
                ret, frame = cap.read()


                if not ret:
                    break
                
                cv2_im = frame
                cv2_im = cv2.flip(cv2_im, 0)
                cv2_im = cv2.flip(cv2_im, 1)

                pil_im = Image.fromarray(cv2_im)
                
                image = pil_im.resize((input_image_size(interpreter)[0:2]), Image.NEAREST)

                im_arr = np.array(image)

                im_arr32 = im_arr.astype(np.float32)

                im_tensor = torch.tensor(im_arr32)
                im_tensor = im_tensor.unsqueeze(0)

                # feed data to input tensor and run the interpreter
                
                interpreter.set_tensor(input_details[0]['index'], im_tensor)
                interpreter.invoke()

                # Obtain results and map them to the classes
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]


                # Get indices of the top k results
                top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

                j=0
                for i in range(top_k_results):

                    pred=predictions[top_k_indices[i]]
                    print(pred ,pred/255.)

                    pred=round(pred,2)
                    lbl=labels[top_k_indices[i]]
                    print(lbl, "=", pred)
                    
                    txt1=lbl + "(" + str(pred) + ")"
                    cv2_im = cv2.rectangle(cv2_im, (25,45 + j*35), (160, 65 + j*35), (0,0,0), -1)
                    cv2_im = cv2.putText(cv2_im, txt1, (30, 60 + j*35),font, 0.5, (255, 255, 255), 1)
                    j=j+1
                    
                pred_max=predictions[top_k_indices[0]]
                lbl_max=labels[top_k_indices[0]]
                                
                #take action based on maximum prediction value
                
                action(pred_max,lbl_max)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2_im = cv2.putText(cv2_im, text_overlay, (60, 30),font, 0.8, (0, 0, 255), 2)
                
                ret, jpeg = cv2.imencode('.jpg', cv2_im)
                pic = jpeg.tobytes()
        
                #Flask streaming
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n')
               
                print("-----------------------------------")
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
        prev_lbl="None"
        app.run(host='127.0.0.1', port=8080, threaded=True) # Run FLASK
        main()
