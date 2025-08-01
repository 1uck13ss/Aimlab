#include <windows.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "3 seconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // current mouse position
    POINT cursorPos;
    GetCursorPos(&cursorPos);

    // new mouse position
    int newX = 10000; 
    int currentY = cursorPos.y; 

    SetCursorPos(newX, currentY); 
/*
    // Simulate left mouse button down and up (fire/click)
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); 

    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));

    std::cout << "Mouse moved and fired!" << std::endl;
*/
    return 0;
}