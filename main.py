import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import odeint

st.title("A Visualisation of the Double Pendulum")
st.sidebar.title("List of Parameters")
time_taken = st.sidebar.slider("time taken for simulation", 0, 100, 10)
m1 = st.sidebar.number_input("input mass 1: ", min_value = 0, value = 1)
m2 = st.sidebar.number_input("input mass 2: ", min_value = 0, value = 1)
l1 = st.sidebar.number_input("input length 1: ", min_value = 0, value = 1)
l2 = st.sidebar.number_input("input length 2: ", min_value = 0, value = 1)
g = st.sidebar.number_input("input gravitational field strength: ", value = 9.8)
initial_theta_1 = np.radians(st.sidebar.number_input("input inital displacement 0f theta 1: "
                                                     , value = 90, min_value = -180
                                                     , max_value = 180 ))
initial_theta_2 = np.radians(st.sidebar.number_input("input inital displacement of theta 2: "
                                                     , value = 120, min_value = -180
                                                     , max_value = 180 ))
initial_vel_theta_1 = st.sidebar.number_input("input inital velocity of theta 1: "
                                              , value = 0)
initial_vel_theta_2 = st.sidebar.number_input("input inital velocity of theta 2: "
                                              , value = 0)
paths = st.sidebar.button("hide paths")
other = st.sidebar.button("hide other trajectory")
dt = 0.02
eps = 0.05
with st.beta_container():
    st.header("Introduction")
    st.markdown("This website is part of our group's M2R research project on the double pendulum at Imperial College London."
                " Group members include Brendan Patalong, Pure Frank, Yian Zeng, James Hyrb and Mingke Peng with project supervisor"
                " Dr Philip Ramsden.")   

# use streamlit functions to take data inputs from user
time_array = np.arange(0, time_taken, dt)
initial1 = np.array([initial_theta_1, initial_theta_2, initial_vel_theta_1,
                    initial_vel_theta_2])
initial2 = np.array([initial_theta_1, initial_theta_2 + eps, initial_vel_theta_1,
                    initial_vel_theta_2])

p = np.array([m1, m2, l1, l2, g])

def derivative(v, t):
    deriv = [0, 0, 0, 0]
    deriv[0] = v[2]
    deriv[1] = v[3]
    A = (p[0] + p[1]) * p[2] ** 2
    B = p[1] * p[2] * p[3] * np.cos(v[0] - v[1])
    D = p[1] * p[3] ** 2
    det = A * D - B ** 2
    f_1 = - p[1] * p[2] * p[3] * v[3] ** 2 * np.sin(v[0] - v[1]) \
          - (p[0] + p[1]) * p[4] * p[2] * np.sin(v[0])
    f_2 = p[1] * p[2] * p[3] * v[2] ** 2 * np.sin(v[0] - v[1]) \
          - p[1] * p[4] * p[3] * np.sin(v[1])
    deriv[2] = det ** (-1) * (D * f_1 - B * f_2)
    deriv[3] = det ** (-1) * (A * f_2 - B * f_1)
    return deriv

soln_1 = odeint(derivative, initial1, time_array)
soln_2 = odeint(derivative, initial2, time_array)

#  convert to cartesian coordinates
x_11 = p[2] * np.sin(soln_1[:, 0])
y_11 = - p[2] * np.cos(soln_1[:, 0])
x_12 = p[3] * np.sin(soln_1[:, 1]) + x_11
y_12 = - p[3] * np.cos(soln_1[:, 1]) + y_11

x_21 = p[2] * np.sin(soln_2[:, 0])
y_21 = - p[2] * np.cos(soln_2[:, 0])
x_22 = p[3] * np.sin(soln_2[:, 1]) + x_21
y_22 = - p[3] * np.cos(soln_2[:, 1]) + y_21

# first figure
fig1 = plt.figure(figsize=(5, 4))
sub = fig1.add_subplot(autoscale_on=False, 
                      xlim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)),
                      ylim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)))
sub.set_aspect('equal')
line1, = sub.plot([], [], "o-", lw=1.7, c = "black")
line2, = sub.plot([], [], "o-", lw=1.7, c = "black")
path1, = sub.plot([], [], ",-", lw=0.8, c = "black")
path2, = sub.plot([], [], ",-", lw=0.8, c = "blue")
line3, = sub.plot([], [], "o-", lw=1.7, c = "blue")
line4, = sub.plot([], [], "o-", lw=1.7, c = "blue")

def realisation(i):
    if not paths:
        path2.set_data(x_12[max(0, i-150):i], y_12[max(0, i-150):i])
        path1.set_data(x_22[max(0, i-150):i], y_22[max(0, i-150):i])

    line1.set_data([0, x_11[i]], [0, y_11[i]])
    line2.set_data([x_11[i], x_12[i]], [y_11[i], y_12[i]])

    if  not other:
        line3.set_data([0, x_21[i]], [0, y_21[i]])
        line4.set_data([x_21[i], x_22[i]], [y_21[i], y_22[i]])
    return path1, path2, line1, line2, line3, line4

# creation of the final animation
animator = animation.FuncAnimation(fig1, realisation, len(soln_1),
                                   interval=dt * 1000, blit=True)

with st.beta_container():
    components.html(animator.to_jshtml(), height=1000)
