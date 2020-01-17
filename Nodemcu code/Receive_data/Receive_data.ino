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

#define LED 4

void setup(){
  Serial.begin(9600);
  start("Niloy","1q2w3e4r5t6Y"); // Wifi details connec to

  pinMode(LED,OUTPUT);
}

void loop(){
  waitUntilNewReq();  //Waits until a new request from python come

   Serial.print(getPath());
  returnThisStr("GotIt");

}
