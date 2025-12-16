from chemlab.util.unit import WAVENUMBER,FREQUENCY,WAVELENGTH,PERIOD
wn = WAVENUMBER(3600, "cm^-1")

freq = wn.convert_to(FREQUENCY).value
wl   = wn.convert_to(WAVELENGTH).value
T    = wn.convert_to(PERIOD).value
print(freq,wl,T)