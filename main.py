import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft, fftfreq

st.title("A Visualisation of the Double Pendulum")
st.sidebar.title("List of Parameters")
angles = st.sidebar.selectbox(label="Angle units", options=("Radians", "Degrees"), index = 0)
m1 = st.sidebar.number_input("input mass 1: ", min_value = 0.00, value = 1.00)
m2 = st.sidebar.number_input("input mass 2: ", min_value = 0.00, value = 1.00)
l1 = st.sidebar.number_input("input length 1: ", min_value = 0.00, value = 1.00)
l2 = st.sidebar.number_input("input length 2: ", min_value = 0.00, value = 1.00)
g = st.sidebar.number_input("input gravitational field strength: ", value = 9.81)
if angles == "Radians":
    initial_theta_1 = st.sidebar.number_input("input inital displacement 0f theta 1: "
                                                     , value = np.pi / 2, min_value = - np.pi
                                                     , max_value = np.pi, step = 1e-6)
    initial_theta_2 = st.sidebar.number_input("input inital displacement 0f theta 2: "
                                                     , value = np.pi * 2 / 3, min_value = - np.pi
                                                     , max_value = np.pi, step = 1e-6)
elif angles == "Degrees":
    initial_theta_1 = np.radians(st.sidebar.number_input("input inital displacement 0f theta 1: "
                                                        , value = 90, min_value = -180
                                                        , max_value = 180 ))
    initial_theta_2 = np.radians(st.sidebar.number_input("input inital displacement of theta 2: "
                                                        , value = 120, min_value = -180
                                                        , max_value = 180 ))
initial_vel_theta_1 = st.sidebar.number_input("input inital velocity of theta 1: "
                                              , value = 0)
initial_vel_theta_2 = st.sidebar.number_input("input initial velocity of theta 2: "
                                              , value = 0)

dt = 0.02
eps = 0.05
initial1 = np.array([initial_theta_1, initial_theta_2, initial_vel_theta_1,
                     initial_vel_theta_2])
initial2 = np.array([initial_theta_1, initial_theta_2 + eps, initial_vel_theta_1,
                     initial_vel_theta_2])
p = np.array([m1, m2, l1, l2, g])

with st.beta_container():
    st.header("Introduction")
    st.markdown("This website is part of our group's M2R research project on the double pendulum at Imperial College London."
                " Group members include B. Patalong, F. Shang, Y. Zeng, J. Hryb and M. Peng with project supervisor"
                " Dr Philip Ramsden. The aim of this project is to discuss further the behaviour of the double pendulum  dynamical system"
                " and use various mathematical and computational methods to determine under what conditions the system becomes chaotic or"
                " displays some form of periodicity.")   
# used pretty much throughout the program
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

# used in phase portrait section
def wrapped(theta, other):
    """Wrap angular displacement to obtain range of -pi to pi for infinite cylinder plot.
    
    Parameters
    ----------
    theta : array
        An output from ODE solver (theta_1 or theta_2)
    other : array
        Other output from ODE solver (omega_1 or omega_2), or time array used

    Returns
    -------
    list
        List containing two lists of lists (for theta, other), split at discontinuities of theta
    """
    
    theta_wrapped = [None] * len(theta)
    for x in enumerate(theta):
        if x[1] > 0:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                 theta_wrapped[x[0]] = x[1] % (2 * np.pi)
        else:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi)
                
    theta_splitted = [] # list of lists
    other_splitted = []
    l = 0
    r = 0
    for y in range(len(theta_wrapped) - 1):
        if abs(theta_wrapped[r] - theta_wrapped[r + 1]) > 0.5:
            theta_splitted.append(theta_wrapped[l:r + 1])
            other_splitted.append(other[l:r + 1])
            l = r + 1
        r += 1

    return [theta_splitted, other_splitted]

# used in phase portrait section
def d_wrapped(theta_1, theta_2):
    """Wrap angular displacements theta_1, theta_2 to obtain range of -pi to pi for torus plot.
    
    Parameters
    ----------
    theta_1 : array
        Displacements of first angle (from ODE solver)
    theta_2 : array
        Displacements of second angle (from ODE solver)
        
    Returns
    -------
    list
        List containing two lists of lists (for theta_1, theta_2), split at discontinuities
    """
    
    theta_1_wrapped = [None] * len(theta_1)
    theta_2_wrapped = [None] * len(theta_2)
    for x in enumerate(theta_1):
        if x[1] > 0:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_1_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                 theta_1_wrapped[x[0]] = x[1] % (2 * np.pi)
        else:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_1_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                theta_1_wrapped[x[0]] = x[1] % (2 * np.pi)
    for x in enumerate(theta_2):
        if x[1] > 0:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_2_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                 theta_2_wrapped[x[0]] = x[1] % (2 * np.pi)
        else:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_2_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                theta_2_wrapped[x[0]] = x[1] % (2 * np.pi)
    
    theta_1_splitted = [] # list of lists
    theta_2_splitted = []
    l = 0
    r = 0
    for y in range(len(theta_1_wrapped) - 1):
        if abs(theta_1_wrapped[r] - theta_1_wrapped[r + 1]) > 0.5 \
        or abs(theta_2_wrapped[r] - theta_2_wrapped[r + 1]) > 0.5:
            theta_1_splitted.append(theta_1_wrapped[l:r + 1])
            theta_2_splitted.append(theta_2_wrapped[l:r + 1])
            l = r + 1
        r += 1
        
    return [theta_1_splitted, theta_2_splitted]

# used in phase portrait section
def wrapped2(theta):
    """Wrap angular displacement to obtain range of -pi to pi for infinite cylinder plot.
    
    Parameters
    ----------
    theta : array
        An output from ODE solver (theta_1 or theta_2)

    Returns
    -------
    list
        List of theta values in range -pi to pi
    """
    
    theta_wrapped = [None] * len(theta)
    for x in enumerate(theta):
        if x[1] > 0:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                 theta_wrapped[x[0]] = x[1] % (2 * np.pi)
        else:
            if (x[1] % (2 * np.pi)) > np.pi:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
            else:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi)
    return theta_wrapped

def doublependpoincare(sol, t):
    
    theta1 = wrapped2(sol[:, 0])
    omega1 = sol[:, 2]
    theta2 = wrapped2(sol[:, 1])
    omega2 = sol[:, 3]
    
    theta_times = [[theta1[i], theta1[i + 1], i, i + 1] for i in range(len(t) - 1) \
                   if (theta1[i] * theta1[i + 1] < 0 and abs(theta1[i]) < 1 and omega1[i] > 0)]
    
    interpolated_theta2 = []
    interpolated_omega2 = []
    
    for m in theta_times:
        interpolated_theta2.append((theta2[m[2]] * m[1] - theta2[m[3]] * m[0]) / (m[1] - m[0]))
        interpolated_omega2.append((omega2[m[2]] * m[1] - omega2[m[3]] * m[0]) / (m[1] - m[0]))
        
    return [interpolated_theta2, interpolated_omega2]

# used in the lyapunov section
# p = np.array([m1, m2, l1, l2, g])
# f = (t1, v1, t2, v2)
def double_pend1(t, w):
    t1, b1, t2, b2 = w
    A = (p[0] + p[1]) * p[2] ** 2
    B = p[1] * p[2] * p[3] * np.cos(t1 - t2)
    D = p[1] * p[3] ** 2
    det = A * D - B ** 2
    f_1 = - p[1] * p[2] * p[3] * b1 ** 2 * np.sin(t1 - t2) \
          - (p[0] + p[1]) * p[4] * p[2] * np.sin(t1)
    f_2 = p[1] * p[2] * p[3] * b2 ** 2 * np.sin(t1 - t2) \
          - p[1] * p[4] * p[3] * np.sin(t2)
    f = [b1,
         det ** (-1) * (D * f_1 - B * f_2),
         b2,
         det ** (-1) * (A * f_2 - B * f_1)]
    return f

# used in lypunov section
def lambda_finder2(x10, y10, x20, y20, t=0.01, p1=0.01, p2=0.01, n=10000):
    x1_old = np.array([x10, y10, x20, y20]) # vector representation of point x1
    X1_old = np.array([x10+p1, y10, x20+p2, y20]) # vector representation of point X1
    lyap_total = 0.0 # initialise total values of lambda
    
    for i in range(n):
        sol1 = solve_ivp(double_pend1, [0, t], x1_old) 
        # take position of x1 after evolving system by a time t
        sol2 = solve_ivp(double_pend1, [0, t], X1_old)
        # take position of X1 after evolving system by a time t
        
        e1 = sol2.y[:,-1] - sol1.y[:, -1]
        # set the error between the new values of x1 and X1
        
        lyap_total += (1/t) * np.log(np.linalg.norm(e1) / np.sqrt(p1**2+p2**2))
        # add the calculated value of lambda to our total
        x1_old = sol1.y[:, -1] # reset and rescale x1, X1, ready for next iteration
        X1_old = sol1.y[:, -1] + e1 / np.linalg.norm(e1) *np.sqrt(p1**2+p2**2)
    return (1/n) * lyap_total

# used in FFT
def wpen(t,y,m1=1,m2=1,a1=1,a2=1,g=9.8):
    
    A  = (m1+m2)*a1**2
    B  = m2*a1*a2*np.cos(y[0]-y[1])
    D  = m2*a2**2
    f1 = -m2*a1*a2*np.sin(y[0]-y[1])*(y[3])**2-a1*(m1+m2)*g*np.sin(y[0])
    f2 = m2*a1*a2*np.sin(y[0]-y[1])*(y[2])**2-m2*a2*g*np.sin(y[1])
    return [y[2],y[3],((1/(A*D-B**2))*(D*f1-B*f2)),((1/(A*D-B**2))*(A*f2-B*f1))]



with st.beta_container():
    st.header("Figures")
    st.markdown("In this section the user can choose to display plots or animations for any given set of parameter values and initial"
                " conditions which can be changed in the sidebar. These plots relate to some of the common computational techniques used for"
                " analysing dynamical systems applied to the double pendulum.")
    dropbox = st.selectbox(label="Select a topic",options=("Double Pendulum Animation", "Linearization about (0,0,0,0)", 
                                "Phase Portraits", "Fast Fourier Transform", "Poincare Sections","Maximal Lyapunov Exponents"), 
                                index = 0)
    
with st.beta_container():
    if dropbox == "Double Pendulum Animation":
        st.markdown("The animation below shows a double pendulum simulation. Note that a second \
                    pendulum is drawn in blue that has initial conditions very close to the ones \
                    given by the user in the sidebar. This allows for a simple demonstration of how \
                    for certain values the dynamical system displays sensitive dependence on initial \
                    conditions or chaotic behaviour.")
        time_taken = st.slider("enter animation run time:", 0, 100, 10)
        # use streamlit functions to take data inputs from user/
        time_array = np.arange(0, time_taken, dt)
        soln_1 = odeint(derivative, initial1, time_array)
        soln_2 = odeint(derivative, initial2, time_array)
        paths = st.button("hide paths")
        other = st.button("hide second trajectory")

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
        fig1 = plt.figure(figsize =(5, 4))
        sub = fig1.add_subplot(autoscale_on=False, 
                            xlim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)),
                            ylim=(- (p[2] + p[3] + 1), (p[2] + p[3] + 1)))
        sub.set_aspect('equal')
        line1, = sub.plot([], [], "o-", lw=1.7, c = "blue")
        line2, = sub.plot([], [], "o-", lw=1.7, c = "blue")
        path1, = sub.plot([], [], ",-", lw=0.8, c = "black")
        path2, = sub.plot([], [], ",-", lw=0.8, c = "blue")
        line4, = sub.plot([], [], "o-", lw=1.7, c = "black")
        line3, = sub.plot([], [], "o-", lw=1.7, c = "black")

        def realisation(i):
            if not paths:
                path2.set_data(x_12[max(0, i-150):i], y_12[max(0, i-150):i])
                path1.set_data(x_22[max(0, i-150):i], y_22[max(0, i-150):i])
            if  not other:
                line3.set_data([0, x_21[i]], [0, y_21[i]])
                line4.set_data([x_21[i], x_22[i]], [y_21[i], y_22[i]])

            line1.set_data([0, x_11[i]], [0, y_11[i]])
            line2.set_data([x_11[i], x_12[i]], [y_11[i], y_12[i]])
        
            return path1, path2, line3, line4, line1, line2

        # creation of the final animation
        animator = animation.FuncAnimation(fig1, realisation, len(y_11),
                                            interval = dt*1000, blit=True)

        components.html(animator.to_jshtml(), height=500)

    elif dropbox == "Linearization about (0,0,0,0)":
        st.markdown("The following figure shows the linearized system about the stable fixed point "r"$(0,0,0,0)$"
                    ". Note this linearization is only an accurate approximation for small angles or for initial"
                    " conditions close to the origin. Also despite this being a linearisation it disaplys some interesting"
                    " behaviour as for different parameter values the solution of the linearized solution can either be quasiperiodic"
                    " or possibly periodic.")
        time_period = st.slider("enter run time: ", 0, 100, 50)
        
        A = np.array([[-p[4] * (p[0] + p[1]) / (p[0] * p[2]), p[4] * p[1]/(p[0] * p[2])],
                      [p[4] * (p[0] + p[1])/(p[0] * p[3]), -p[4] * (p[1] + p[0])/(p[0] * p[3])]])
        ev = np.linalg.eig(A)
        coefs1 = np.linalg.solve(ev[1], np.array([initial1[0],initial1[1]]))
        coefs2 = np.linalg.solve(ev[1], np.array([initial1[2],initial1[3]]))
        lamb = np.sqrt(-ev[0])
        t = np.linspace(0, time_period, 10001)
        theta1 = coefs1[0] * ev[1][0][0] * np.cos(lamb[0] * t) + coefs1[1] * ev[1][0][1] * np.cos(lamb[1] * t) +\
                 coefs2[0] * ev[1][0][0] * np.sin(lamb[0] * t) + coefs2[1] * ev[1][0][1] * np.sin(lamb[1] * t)
        theta2 = coefs1[0] * ev[1][1][0] * np.cos(lamb[0] * t) + coefs1[1] * ev[1][1][1] * np.cos(lamb[1] * t) + \
                 coefs2[0] * ev[1][1][0] * np.sin(lamb[0] * t) + coefs2[1] * ev[1][1][1] * np.sin(lamb[1] * t)
        
        fig_linear = plt.figure(figsize = (5, 4))
        plt.plot(t, theta1, label=r"$\theta_1$")
        plt.plot(t, theta2, label=r"$\theta_2$")
        plt.xlabel("Time, t")
        plt.ylabel("Angular Displacements")
        plt.legend()
        st.pyplot(fig_linear)

    elif dropbox == "Phase Portraits":
        st.markdown("The below plots show various phase portraits in 2D for the given initial conditions and parameters.")
        time_taken = st.slider("Total Time", 100, 1000, 100)
        t = np.linspace(0, time_taken, 10001)
        sol = odeint(derivative, initial1, t)

        splitted_1 = wrapped(sol[:, 0], t)
        splitted_2 = wrapped(sol[:, 0], sol[:, 2])
        splitted_3 = wrapped(sol[:, 1], t)
        splitted_4 = wrapped(sol[:, 1], sol[:, 3])

        fig2 = plt.figure()
        if splitted_1[0]: # if else to check for wrapping around - if none, plot original outputs as normal
            for i in range(len(splitted_1[0])):
                plt.plot(splitted_1[1][i], splitted_1[0][i])
        else:
            plt.plot(t, sol[:, 0])
        plt.xlabel("time")
        plt.ylabel(r"$\theta_1$")

        fig3 = plt.figure()
        if splitted_2[0]: 
            for i in range(len(splitted_2[0])):
                plt.plot(splitted_2[0][i], splitted_2[1][i])
        else:
            plt.plot(sol[:, 0], sol[:, 2])
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\dot\theta_1$")

        fig4 = plt.figure()
        if splitted_3[0]:
            for i in range(len(splitted_3[0])):
                plt.plot(splitted_3[1][i], splitted_3[0][i])
        else:
            plt.plot(t, sol[:, 1])
        plt.xlabel("time")
        plt.ylabel(r"$\theta_2$")

        fig5 = plt.figure()
        if splitted_4[0]:
            for i in range(len(splitted_4[0])):
                plt.plot(splitted_4[0][i], splitted_4[1][i])
        else:
            plt.plot(sol[:, 1], sol[:, 3])
        plt.xlabel(r"$\theta_2$")
        plt.ylabel(r"$\dot\theta_2$")

        ##########################################

        splitted = d_wrapped(sol[:, 0], sol[:, 1])

        fig6 = plt.figure()
        if splitted[0] and splitted[1]:
            for i in range(len(splitted[0])):
                plt.plot(splitted[0][i], splitted[1][i])
        else:
            plt.plot(sol[:, 0], sol[:, 1])
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")

        st.pyplot(fig2)
        st.pyplot(fig3)
        st.pyplot(fig4)
        st.pyplot(fig5)
        st.pyplot(fig6)
        
    elif dropbox == "Poincare Sections":
        st.markdown("Poincare sections are a useful tool for visualising possibly chaotic behaviour. \
                     The double pendulum system is 4-dimensional which is difficult to visualise, however \
                     using poincare sections allows us to analyse the system in less dimesnions by reducing \
                     the system to a discrete time series in fewer dimensions.")
        time_taken = st.slider("Total Time", 5000, 50000, 5000)
        t = np.linspace(0, time_taken, 100001)
        sol = odeint(derivative, initial1, t)

        x = doublependpoincare(sol, t)[0]
        y = doublependpoincare(sol, t)[1]

        fig_poincare = plt.figure(figsize=(10, 10))
        plt.scatter(x, y, s=1, c="m")
        plt.xlabel(r"$\theta_2$")
        plt.ylabel(r"$\dot\theta_2$")
        st.pyplot(fig_poincare)

    elif dropbox == "Maximal Lyapunov Exponents":
        st.markdown("The Maximal Lyapunov Exponent, MLE is an asymptotic measure and so can only be approximated using computational methods.\
                     If the value of the MLE is close zero then this would suggest nonchaotic behaviour whereas larger values\
                     of the MLE correspond to chaotic behaviour.")
        MLA = lambda_finder2(initial_theta_1, initial_vel_theta_1,
                             initial_theta_2, initial_vel_theta_2)
        st.markdown("The MLA for the given initial conditions and parameters is " +str(MLA))
    
    elif dropbox == "Fast Fourier Transform":
        st.markdown("The Fast Fourier Transform , FFT is a commonly used method of analysis \
                     for dynamical systems and signal processing more generally. \
                     The fast fourier transform is an algorithm that can be applied to discrete data to \
                     find the discrete fourier transform of the data. The uses of the FFT, relating to \
                     the double pendulum are discussed further in our research paper.")
        sols = solve_ivp(wpen, [0,1000], initial1, max_step =0.05, m1 = p[0], 
        m2 = p[1], l1=p[2], l2=p[3], g=p[4])
        L=len(sols.t)
        L = len(sols.t) # you need this, which is the size of the data set
        T = 2000/(len(sols.y[0])-1) # or just set this equal to max_step
        y = sols.y[0]
        yf = fft(y)
        xf = fftfreq(L, T)[:2000//2] # note first argument is L, not N
    
        fig8, ax = plt.subplots(figsize=(10,10))
        ax.plot(xf*2*np.pi, (2/2000)*np.abs(yf[:2000//2]),'r')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        # is 2/N the right scale factor? doesn't matter much I guess
        ax.grid()
        st.pyplot(fig8)
