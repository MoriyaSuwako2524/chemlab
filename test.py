import numpy as np
#import matplotlib
#matplotlib.use("Qt5Agg")
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import  Figure
import matplotlib.pyplot as plt


plt.style.use("ggplot")
import scipy.special as spe
import scipy.constants as const

a0 = const.value(u"Bohr radius")



def psi_R(r, n=1, l=0):
    coeff = np.sqrt(
        (2.0 / (a0 * n)) ** 3 *
        spe.factorial(n - l - 1) / (2.0 * (n) * spe.factorial(n + l)))
    laguerre = spe.assoc_laguerre(2.0 * r / (a0 * n), n - l - 1, 2 * l + 1)
    return coeff * np.exp(-r / (a0 * n)) * (2.0 * r / (a0 * n)) ** l * laguerre

def draw_1():



    n = 2
    l = 0
    # m = 0

    r = np.linspace(0, (5 - n) * a0 * n ** 2, 200)

    fig = plt.figure()
    ax = fig.gca()
    for l in range(n):
        ax.set_title("Radial Probability Distribution n = %d" % (n))
        vr = psi_R(r, n, l) ** 2 * (r ** 2)
        ax.plot(r, vr, label="l = %d" % l)
    ax.legend(loc="best")
    # plt.savefig("./radial_probability_distribution_n=%d.jpg" % n)
    plt.show()


def draw_2():


    n = 4
    l = 0
    i = 1
    # m = 0

    r = np.linspace(0, 4 * a0 * n ** 2, 200)

    # fig =
    plt.figure()
    # ax = fig.gca()
    for i in range(n):
        # j = i+1
        # for j in range(n):
        ax = plt.subplot(n, n, i + 1)
        # ax.set_title("Radial Probability Distribution n = %d" % (n))
        vr = psi_R(r, n, i) ** 2 * (r ** 2)

        ax.plot(r, vr, label="l = %d" % l)
    # ax.legend(loc="best")
    # plt.savefig("./radial_probability_distribution_n=%d.jpg" % n)
    plt.show()


def draw_3():
    n = 4
    l = 0

    # m = 0

    r = np.linspace(0, 4 * a0 * n ** 2, 200)

    # fig =
    plt.figure()
    # ax = fig.gca()
    for i in range(n):
        # j = i+1
        for j in range(n):
            ax = plt.subplot(n, n, n * i + j + 1)
            # ax.set_title("Radial Probability Distribution n = %d" % (n))
            vr = psi_R(r, i + 1, j) ** 2 * (r ** 2)

            ax.plot(r, vr, label="l = %d" % l)
    # ax.legend(loc="best")
    # plt.savefig("./radial_probability_distribution_n=%d.jpg" % n)
    plt.show()


def draw_4():

    n = 4
    l = 0

    # m = 0


    # fig =
    plt.figure(figsize=[15, 9])
    # ax = fig.gca()
    for i in range(n):
        r = np.linspace(0, 3.5 * a0 * (i + 1) ** 2, 200)
        for j in range(i + 1):
            ax = plt.subplot(n, n, n * i + j + 1)
            # ax.set_title("Radial Probability Distribution n = %d" % (n))
            vr = psi_R(r, i + 1, j) ** 2 * (r ** 2)

            ax.plot(r, vr, label="l = %d" % l)
            ax.legend(loc="best")
    # plt.savefig("./radial_probability_distribution_n=%d.jpg" % n)
    plt.suptitle("MY SHOP")
    plt.show()

