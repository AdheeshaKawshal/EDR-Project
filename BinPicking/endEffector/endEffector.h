/*
 * endEffector.h
 *
 * Created: 7/3/2025 8:59:28 PM
 *  Author: HP VICTUS
 */ 


#ifndef ENDEFFECTOR_H_
#define ENDEFFECTOR_H_

#include <stdint.h>
// PID constants (tune as needed)


void activate_gripper();
void lower_gripper();
void lift_gripper();
void release_gripper();
void move_gripper(string dir);//up, down

#endif /* ENDEFFECTOR_H_ */