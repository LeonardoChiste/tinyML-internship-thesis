# tinyML-internship-thesis
This is a repository for the internship and thesis of Leonardo Chistè in embedded machine learning
### KWS on Arduino nano 33 ble
Blink the red LED when "hello world" is pronounced. https://studio.edgeimpulse.com/studio/746329

* Hardware requirements: Arduino Nano 33 BLE
* Software requirements: Arduino ide with driver library for the board: Arduino Mbed os nano board 
* Download the library .zip from the /kws_helloworld folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload

### Speaker Verification on Arduino nano 33 ble
The model is trained on examples of me, not me and noise where the "me" examples are any words/sentences. The .ino is an example built into the library. 
https://studio.edgeimpulse.com/studio/750711

* Hardware requirements: Arduino Nano 33 BLE
* Software requirements: Arduino ide with driver library for the board: Arduino Mbed os nano board 
* Download the library .zip from the /sv folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload

### Speaker Verification Helloworld on Arduino nano 33 ble
The model is trained on examples of me, not me and noise where the "me" and "not me" examples are only people saying "helloworld", This is to highlight the pith differences on a single word. The .ino is an example built into the library. 
https://studio.edgeimpulse.com/studio/752142

* Hardware requirements: Arduino Nano 33 BLE
* Software requirements: Arduino ide with driver library for the board: Arduino Mbed os nano board 
* Download the library .zip from the /sv_helloworld folder
* On Arduino IDE: Schetch -> Include library -> Add .ZIP library...
* Download the .ino file, compile and upload
