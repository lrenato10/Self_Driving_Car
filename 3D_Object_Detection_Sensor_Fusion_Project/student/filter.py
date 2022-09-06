# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import misc.params as params

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        return np.matrix([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q = params.q
        dt = params.dt
        qdt3 = ((dt**3)/3) * q 
        qdt2 = ((dt**2)/2) * q 
        qdt = dt * q 
        return np.matrix([[qdt3,    0,      0,      qdt2,   0,      0   ],
                          [0,       qdt3,   0,      0,      qdt2,   0   ],
                          [0,       0,      qdt3,   0,      0,      qdt2],
                          [qdt2,    0,      0,      qdt,    0,      0   ],
                          [0,       qdt2,   0,      0,      qdt,    0   ],
                          [0,    0,         qdt2,   0,      0,      qdt ]])
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        Q = self.Q()
        x = track.x
        P = track.P

        x = F*x
        P = F*P*F.transpose() + Q

        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        P=track.P
        x=track.x
        I = np.identity(params.dim_state)
        H = meas.sensor.get_H(x) # H(x)=Hx
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)

        K = P*H.transpose()*np.linalg.inv(S) #Kalman Gain
        x = x + K*gamma
        P = (I - K*H)*P

        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        x = track.x
        Hx = meas.sensor.get_hx(x) #non linear due to camera model
        z = meas.z
        return z - Hx
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        P = track.P
        R = meas.R
        return H*P*H.transpose() + R
        ############
        # END student code
        ############ 