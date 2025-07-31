/*
 * endEffector.c
 *
 * Created: 7/3/2025 8:59:50 PM
 *  Author: HP VICTUS
 */ 

#inlcude "ENDEFFECTOR_DIR2.h"
#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <stdint.h>

#define F_CPU 16000000UL  // CPU frequency (16 MHz)

#define ENDEFFECTOR_DIR1    DDD5
#define ENDEFFECTOR_DIR2    DDB6
#define ENDEFFECTOR_PWM     DDB7
#define ENCODER_A           DDD0

#define VACUM_PUMP          DDC7
#define SOLENOID_VALVE      DDE2

#define Kp  1.5
#define Ki  0.05
#define Kd  0.1

// Motion planning
#define MAX_SPEED 200         // Max PWM
#define ACCEL_DIST 200        // Pulse count used for acceleration
#define DECEL_DIST 200        // Pulse count used for deceleration

volatile long pulse_count = 0;
volatile long target_count = 1000;

volatile int16_t error = 0;
volatile long integral = 0;
volatile int16_t last_error = 0;
volatile int16_t derivative = 0;


// Encoder pulse ISR
ISR(INT0_vect) {
    pulse_count++;
}

void endeffector_init() {
    DDRD &= ~(1 << PD2);  // PD2 as input
    EIMSK |= (1 << INT0); // Enable INT0
    EICRA |= (1 << ISC00); // Any edge triggers interrupt
    DDRD |= (1 << PD6) | (1 << PD7); // PD6 = PWM, PD7 = Direction
    TCCR0A = (1 << COM0A1) | (1 << WGM01) | (1 << WGM00); // Fast PWM, non-inverting
    TCCR0B = (1 << CS01); // Prescaler = 8
    OCR0A = 0; // Start stopped
}

void set_motor_speed(int16_t speed) {
    if (speed < 0) {
        PORTD &= ~(1 << PD7); // DIR = 0 → Reverse
        speed = -speed;
    } else {
        PORTD |= (1 << PD7);  // DIR = 1 → Forward
    }

    if (speed > 255) speed = 255;
    OCR0A = speed;
}

uint8_t s_curve_limit_speed(long current_pos, long total_distance) {
    long remaining = total_distance - current_pos;

    if (current_pos <= ACCEL_DIST) {
        // Accelerating phase
        return (uint8_t)((float)current_pos / ACCEL_DIST * MAX_SPEED);
    } else if (remaining <= DECEL_DIST) {
        // Decelerating phase
        return (uint8_t)((float)remaining / DECEL_DIST * MAX_SPEED);
    } else {
        // Cruising phase
        return MAX_SPEED;
    }
}
uint8_t s_curve_limit_speed(long current_pos, long total_distance) {
    long remaining = total_distance - current_pos;
    float x, speed;

    if (current_pos < ACCEL_DIST) {
        // Smooth acceleration (cubic ease-in)
        x = (float)current_pos / ACCEL_DIST;
        speed = MAX_SPEED * (3 * x * x - 2 * x * x * x);
    } else if (remaining < DECEL_DIST) {
        // Smooth deceleration (cubic ease-out)
        x = (float)remaining / DECEL_DIST;
        speed = MAX_SPEED * (3 * x * x - 2 * x * x * x);
    } else {
        // Constant speed
        speed = MAX_SPEED;
    }

    if (speed < 0) speed = 0;
    if (speed > 255) speed = 255;  // Clamp for PWM

    return (uint8_t)speed;
}

void move_gripper(string dir){
        while (pulse_count < target_count) {
        // PID control
        error = target_count - pulse_count;

        integral += error;
        if (integral > 10000) integral = 10000;
        else if (integral < -10000) integral = -10000;

        derivative = error - last_error;
        last_error = error;

        int32_t raw_output = Kp * error + Ki * integral + Kd * derivative;

        // S-curve speed limiting
        uint8_t limit = s_curve_limit_speed(pulse_count, target_count);

        if (raw_output > limit) raw_output = limit;
        if (raw_output < -limit) raw_output = -limit;

        set_motor_speed((int16_t)raw_output);

        _delay_ms(10);  // Control loop period
    }
    set_motor_speed(0);
}
