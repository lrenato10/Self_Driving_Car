/**********************************************
 * Self-Driving Car Nano-degree - Udacity
 *  Created on: December 11, 2020
 *      Author: Mathilde Badoual
 **********************************************/

#include "pid_controller.h"
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kpi, double Kii, double Kdi, double output_lim_maxi, double output_lim_mini) {
   /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   **/
	this->Kp=Kpi;
  	this->Ki=Kii;
  	this->Kd=Kdi;
  	this->out_max=output_lim_maxi;
  	this->out_min=output_lim_mini;
  	this->err=0;
  	this->diff_err=0;
  	this->int_err=0;
  	this->prev_err=0;
  	this->dt=0;
  
}


void PID::UpdateError(double cte) {
   /**
   * TODO: Update PID errors based on cte.
   **/
	this->err=cte;
  	this->diff_err=(cte-this->prev_err)/this->dt;
  	this->int_err+=cte*this->dt;
  	this->prev_err=cte;
}

double PID::TotalError() {
   /**
   * TODO: Calculate and return the total error
    * The code should return a value in the interval [output_lim_mini, output_lim_maxi]
   */
   	double control;
  	// calcul control action
  	control = this->Kp * this->err + this->Ki * this->int_err + this->Kd * this->diff_err;
    // saturation (out_max <= control <= out_max)
  	control = std::max(control, this->out_min);
    control = std::min(control, this->out_max);
   
   	return control;
}

double PID::UpdateDeltaTime(double new_delta_time) {
   /**
   * TODO: Update the delta time with new value
   */
  	this->dt=new_delta_time;
}