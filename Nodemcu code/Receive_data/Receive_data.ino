/* LED CONTROLLING WITH PYTHON
 * Written by Junicchi
 * https://github.com/Kebablord 
 *
 * It's a ESP management through Python example
 * It simply fetches the path from the request
 * Path is: https://example.com/this -> "/this"
 * You can command your esp through python with request paths
 * You can read the path with getPath() function
 */


#include "ESP_MICRO.h"



//motorPins
const int motorPin1 = 14,motorPin2 = 12;        //right motor
const int motorPin3 = 13,motorPin4 = 15;       //left motor

int rotateDelay=100, forwardDelay=1000;
void setup(){
  Serial.begin(9600);
  start("Epitapher Thanda","Qawsedrf"); // Wifi details connec to
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorPin3, OUTPUT);
  pinMode(motorPin4, OUTPUT);
}

void loop(){
  waitUntilNewReq();  //Waits until a new request from python come

  Serial.print(getPath());
  if (getPath()=="/nn")
  {
    goForward(forwardDelay);
    brake();
  }

  else if (getPath()=="/ww")
  {
     rotateLeft(rotateDelay);
     goForward(forwardDelay);
    brake();
  }


  else if (getPath()=="/ee")
  {
     rotateRight(rotateDelay);
     goForward(1000);
     brake();
  }
  
  returnThisStr("GotIt");
  goForward(forwardDelay);


}


void goForward(int delayTime)
{
  analogWrite(motorPin1, 255);
  analogWrite(motorPin2 , 0);
  analogWrite(motorPin3,255);
  analogWrite(motorPin4, 0);
  delay(delayTime);
  

}

void rotateLeft(int delayTime)
{
  analogWrite(motorPin1, 200);
  analogWrite(motorPin2 , 0);
  analogWrite(motorPin3, 0);
  analogWrite(motorPin4, 200);
  delay(delayTime);
  

}

void rotateRight(int delayTime)
{
  analogWrite(motorPin1, 0);
  analogWrite(motorPin2 , 200);
  analogWrite(motorPin3, 200);
  analogWrite(motorPin4, 0);
  delay(delayTime);
  

}


void brake(int delayTime)
{
  analogWrite(motorPin1, 0);
  analogWrite(motorPin2 , 0);
  analogWrite(motorPin3, 0);
  analogWrite(motorPin4, 0);
  delay(delayTime);

}



