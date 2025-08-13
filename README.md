# tinyML-internship-thesis
This is a repository for the internship and thesis of Leonardo Chistè in embedded machine learning
* Hardware requirements: Arduino Nano 33 BLE
* Software requirements: Arduino ide with driver library for the board: Arduino Mbed os nano board 
### KWS on Arduino nano 33 ble
Blink the red LED when "hello world" is pronounced. https://studio.edgeimpulse.com/studio/746329
* Download the library .zip from the /kws_helloworld folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload

### Speaker Verification on Arduino nano 33 ble
The model is trained on examples of me, not me and noise where the "me" examples are any words/sentences. The .ino is an example built into the library. 
https://studio.edgeimpulse.com/studio/750711

* Download the library .zip from the /sv folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload

### Speaker Verification Helloworld on Arduino nano 33 ble
The model is trained on examples of me, not me and noise where the "me" and "not me" examples are only people saying "helloworld", This is to highlight the pith differences on a single word. The .ino is an example built into the library. 
https://studio.edgeimpulse.com/studio/752142

* Download the library .zip from the /sv_helloworld folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload

### D-vector 32 speaker verification
The raw data samples are normalized, passed through a mfcc function and a CNN that produces a d-vector of 32 float values, which is compared with a stored d-vector to determine if I am speaking.
The training script is included in the /training folder. The data set is composed of about 600 1 second samples from 6 speakers(including me) and noise. 

* Download the tflite micro library .zip from https://github.com/tensorflow/tflite-micro-arduino-examples
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Open the svModel folder in Arduino IDE and upload the code

