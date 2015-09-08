import math
import sys
import ode
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt

def fprime(t, x, *args, **kwargs):
    deriv = x.copy()
    i = len(x)/2
    M = args[0]
    
    dx = x[:i:2,None]-x[None,:i:2]
    dy = x[1:i:2,None]-x[None,1:i:2]
    r = np.sqrt(dx*dx + dy*dy)
    deriv[:i] = x[i:]
    inds = np.arange(len(x)/4)
    for j in xrange(len(x)/4):
        notj = inds != j
        ir = 1.0 / r[j,notj]
        deriv[i+2*j] = (-M[notj] * dx[j,notj] * ir*ir*ir).sum()
        deriv[i+2*j+1] = (-M[notj] * dy[j,notj] * ir*ir*ir).sum()
    return deriv

def dTdP(t, q, p, *args, **kwargs):
    M = args[0]
    return p / np.repeat(M,2)

def dVdQ(t, q, p, *args, **kwargs):
    
    dx = q[ ::2,None]-q[None, ::2]
    dy = q[1::2,None]-q[None,1::2]
    r = np.sqrt(dx*dx + dy*dy)
    
    inds = np.arange(len(q)/2)
    f = np.zeros(p.shape)
    for i in xrange(len(f)/2):
        noti = (inds != i)
        ir = 1.0/r[i,noti]
        f[2*i  ] = (M[noti] * dx[i,noti]*ir*ir*ir).sum()
        f[2*i+1] = (M[noti] * dy[i,noti]*ir*ir*ir).sum()

    return f

def evolve_rk(x0, t0, t1, n, step, *args, **kwargs):

    t = np.linspace(t0, t1, num=n+1)
    dt = (t1-t0)/n

    x = np.zeros((n+1,len(x0)))
    x[0,:] = x0

    for i in xrange(n):
        x[i+1] = step(t[i], x[i], fprime, dt, *args, **kwargs)

    return t, x

def evolve_symp(q0, p0, t0, t1, n, step, *args, **kwargs):

    t = np.linspace(t0, t1, num=n+1)
    dt = (t1-t0)/n

    q = np.zeros((n+1,len(q0)))
    p = np.zeros((n+1,len(p0)))
    q[0] = q0
    p[0] = p0
    
    for i in xrange(n):
        q[i+1], p[i+1] = step(t[i], q[i], p[i], dTdP, dVdQ, dt, *args, **kwargs)

    return t, q, p

def energy(q, p, M):

    mass = np.repeat(M,2)
    T = 0.5*(p*p/mass).sum()

    dx = q[::2,None]-q[None,::2]
    dy = q[1::2,None]-q[None,1::2]
    r = np.sqrt(dx*dx + dy*dy)
    
    inds = np.arange(len(q)/2)

    V = 0.0
    for j in xrange(len(q)/2):
        notj = inds != j
        ir = 1.0 / r[j,notj]
        V += (-M[notj] * ir).sum()
    V *= 0.5

    return T + V

def plot_bodies(fig, t, q, p, M, *args, **kwargs):
    plt.figure(fig.number)
    plt.subplot(2,2,1)
    for i in xrange(q.shape[1]/2):
        plt.plot(q[:,2*i], q[:,2*i+1], *args, **kwargs)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.subplot(2,2,2)
    en = np.zeros(t.shape)
    for i in xrange(len(t)):
        en[i] = energy(q[i], p[i], M)
    plt.plot(t, en, *args, **kwargs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$E$')
    plt.subplot(2,2,3)
    plt.plot(t, np.sqrt((p[:,::2]*p[:,::2]+p[:,1::2]*p[:,1::2]).sum(axis=1)), *args, **kwargs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    plt.subplot(2,2,4)
    plt.plot(t, (q[:,::2]*p[:,1::2]-q[:,1::2]*p[:,::2]).sum(axis=1), *args, **kwargs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$J_z$')
    plt.tight_layout()

    
if len(sys.argv) < 4:
    print("\nusage: python bodies.py nb n T <schemes...>\n")
    print("nb: Number of bodies (2 or 3)")
    print("n:  Number of time steps")
    print("T:  Total integration time")
    print("schemes: which numerical schemes to compare")
    print("    - fe:  Forward Euler")
    print("    - rk2: Runge-Kutta Second Order")
    print("    - rk4: Runge-Kutta Fourth Order")
    print("    - s1:  Symplectic First Order")
    print("    - s2:  Symplectic Second Order")
    print("    - s4:  Symplectic Fourth Order")
    print("\nexample: python bodies.py 3 1000 5 fe rk2 rk4\n")
    sys.exit()

t0 = 0.0

nb = int(sys.argv[1])
n = int(sys.argv[2])
t1 = float(sys.argv[3])

if nb == 3:
    M = np.array([1.0,1.0,1.0])
    x = np.array([0.97000436,-0.24308753])
    v = np.array([-0.93240737,-0.86473146])
    q0 = np.concatenate((x, -x, [0.0,0.0]))
    p0 = np.concatenate((-0.5*v, -0.5*v, v))

else:
    nb = 2
    M = np.array([1.0,1.0])
    q0 = np.array([0.5,0.0, -0.5,0.0])
    p0 = np.array([0.0,0.71,0.0,-0.71])

x0 = np.concatenate((q0,p0))

fig = plt.figure(figsize=(12,9))

if "fe" in sys.argv:
    print("Forward Euler...")
    t, x = evolve_rk(x0, t0, t1, n, ode.forward_euler, M)
    print("   Plotting...")
    plot_bodies(fig, t, x[:,:2*nb], np.repeat(M,2)*x[:,2*nb:], M, 'k')

if "rk2" in sys.argv:
    print("RK2...")
    t, x = evolve_rk(x0, t0, t1, n, ode.rk2, M)
    print("   Plotting...")
    plot_bodies(fig, t, x[:,:2*nb], np.repeat(M,2)*x[:,2*nb:], M, 'b')

if "rk4" in sys.argv:
    print("RK4...")
    t, x = evolve_rk(x0, t0, t1, n, ode.rk4, M)
    print("   Plotting...")
    plot_bodies(fig, t, x[:,:2*nb], np.repeat(M,2)*x[:,2*nb:], M, 'g')

if "s1" in sys.argv:
    print("Symplectic 1...")
    t, q, p = evolve_symp(q0, p0, t0, t1, n, ode.symp1, M)
    print("   Plotting...")
    plot_bodies(fig, t, q, p, M, 'y')

if "s2" in sys.argv:
    print("Symplectic 2...")
    t, q, p = evolve_symp(q0, p0, t0, t1, n, ode.symp2, M)
    print("   Plotting...")
    plot_bodies(fig, t, q, p, M, 'r')

if "s4" in sys.argv:
    print("Symplectic 4...")
    t, q, p = evolve_symp(q0, p0, t0, t1, n, ode.symp4, M)
    print("   Plotting...")
    plot_bodies(fig, t, q, p, M, 'k')

if "anim" in sys.argv:

    anim_fig = plt.figure()
    line, = plt.plot([], [], 'r-', alpha=0.5)
    point, = plt.plot([], [], 'ro')
    plt.xlim(Q.min(), Q.max())
    plt.ylim(P.min(), P.max())
    plt.xlabel(r'$q(t)$')
    plt.ylabel(r'$p(t)$')

    
    def animate(i):
        line.set_data(Q[:i+1], P[:i+1])
        point.set_data(Q[i], P[i])

    ani = anim.FuncAnimation(anim_fig, animate, len(T), interval=50)
    ani.save("oscillate.mp4")

plt.show()
