import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

dic, data = ng.fileio.agilent.read_fid("D2O_fid", as_2d=True)
params = ng.fileio.agilent.read_procpar("D2O_procpar")




acq_time = float(params['at']['values'][0])
time_per_point = acq_time/(data.shape[1])
time=np.arange(0,acq_time,time_per_point)
print("The FID acquisition took " + "{:1.5}".format(acq_time) + " seconds.")






fftdata = np.fft.fft(data)

NMR_freq = 400
freq = np.fft.fftfreq(time.shape[0],time_per_point)
chem_shift = -(freq - 1986)/NMR_freq

theta = np.degrees(10)
plt.plot(chem_shift,fftdata.real[0]*np.cos(np.radians(theta)) + fftdata.imag[0]*np.sin(np.radians(theta)))
ax = plt.gca()
ax.set_xlim(-3,10)
ax.set_xlim(ax.get_xlim()[::-1])
ax.axes.yaxis.set_ticklabels([])
ax.axes.yaxis.set_ticks([])
plt.title("Figure 6 - NMR spectrum")
plt.xlabel("Chemical shift (ppm)")
plt.show()