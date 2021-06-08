import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import odeint

# use streamlit functions to \take data in from the user
st.title("A Visualisation of the Double Pendulum")
st.sidebar.title("List of Parameters")
time_taken = st.sidebar.slider("time taken for simulation", 0, 100, 50)
m1 = st.sidebar.number_input("input mass 1: ", min_value = 0, value = 1)
m2 = st.sidebar.number_input("input mass 2: ", min_value = 0, value = 1)
l1 = st.sidebar.number_input("input length 1: ", min_value = 0, value = 1)
l2 = st.sidebar.number_input("input length 2: ", min_value = 0, value = 1)
g = st.sidebar.number_input("input gravitational field strength: ", value = 9.8)
initial_theta_1 = np.radians(st.sidebar.number_input("input inital displacement 0f theta 1: "
                                                     , value = 90))
initial_theta_2 = np.radians(st.sidebar.number_input("input inital displacement of theta 2: "
                                                     , value = 90))
initial_vel_theta_1 = st.sidebar.number_input("input inital velocity of theta 1: "
                                              , value = 0)
initial_vel_theta_2 = st.sidebar.number_input("input inital velocity of theta 2: "
                                              , value = 0)
paths = st.sidebar.button("hide paths")
dt = 0.02

initial = np.array([initial_theta_1, initial_theta_2, initial_vel_theta_1,
                    initial_vel_theta_2])
p = np.array([m1, m2, l1, l2, g])
time_array = np.arange(0, time_taken, dt)


def derivative(v, t):
    """
    Calculates the LHS of the double pendulum

    Parameters
    ----------
    v : array
        An np array containing the 4 variables often denoted
        theta_1, theta_2, omega_1, omega_2
    t : array
        Time array used when solving using odeint later
    p : array
        An np array containing the parameters m1, m2, l1, l2, g
    """
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

# solves the above system numerically
soln = odeint(derivative, initial, time_array)

#  convert to cartesian coordinate
x_1 = p[2] * np.sin(soln[:, 0])
y_1 = - p[2] * np.cos(soln[:, 0])
x_2 = p[3] * np.sin(soln[:, 1]) + x_1
y_2 = - p[3] * np.cos(soln[:, 1]) + y_1

fig = plt.figure(figsize=(5, 4))
sub = fig.add_subplot(autoscale_on=False, 
                      xlim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)),
                      ylim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)))
sub.set_aspect('equal')
line1, = sub.plot([], [], "o-", lw=1.7, c="black")
line2, = sub.plot([], [], "o-", lw=1.7, c="black")
path1, = sub.plot([], [], ",-", lw=0.8, c ="red")
path2, = sub.plot([], [], ",-", lw=0.8, c ="blue")

def realisation(i):
    """
    Returns the pendulum and its previous path at a given time
    Note this function will be passed as a parameter of a 
    FuncAnimation class which will then produce the animation.
    params:
    ----
    i is the given time instance
    """
    if not paths:
        path2.set_data(x_2[max(0, i-150):i], y_2[max(0, i-150):i])
        path1.set_data(x_1[max(0, i-150):i], y_1[max(0, i-150):i])
    line1.set_data([0, x_1[i]], [0, y_1[i]])
    line2.set_data([x_1[i], x_2[i]], [y_1[i], y_2[i]])
    return path1, path2, line1, line2
# creation of the final animation
animator = animation.FuncAnimation(fig, realisation, len(soln),
                                   interval=dt * 1000, blit=True)
components.html(animator.to_jshtml(), height=1000)
