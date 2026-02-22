import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from solargeorisk_extension.data_prep import load_data_intertemporal_from_excel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = load_data_intertemporal_from_excel(os.path.join(ROOT, "inputs", "input_data_intertemporal.xlsx"))
print("regions:", d.regions)
print("players:", d.players)
print("times:", d.times)
print("Dmax_t sample:", list(d.Dmax_t.items())[:6])
print("Kcap_2025:", d.Kcap_2025)
print("s_ub:", d.s_ub)
print("f_hold:", d.f_hold)
print("c_inv:", d.c_inv)
print("Data load OK")
