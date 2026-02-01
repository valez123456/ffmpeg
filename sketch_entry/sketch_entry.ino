#include <Mouse.h>

void setup() {
  Serial.begin(115200);
  Mouse.begin();
}

void loop() {
  if (Serial.available() >= 2) {
    int8_t dx = (int8_t)Serial.read();
    int8_t dy = (int8_t)Serial.read();
    Mouse.move(dx, dy);
  }
}
