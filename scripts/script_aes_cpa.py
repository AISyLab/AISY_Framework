import aisy_sca
from app import *
import matplotlib.pyplot as plt

aisy = aisy_sca.AisyCPA()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["hw_aes.h5"])
target_byte = 0
aisy.set_aes_leakage_model(leakage_model="HW",
                           byte=target_byte,
                           direction="Encryption",
                           cipher="AES128",
                           round=10,
                           target_state="Sbox",
                           attack_direction="output")
key_rank_report_interval = 1000
correlation, key_rank_evolution_cpa = aisy.run(key_rank_report_interval=key_rank_report_interval)
correlation = correlation.T

known_key = datasets_dict["hw_aes.h5"]["key"]
known_key_int = [int(x) for x in bytearray.fromhex(known_key)]

plt.subplot(1, 2, 1)
for kg in range(256):
    if kg != known_key_int[target_byte]:
        plt.plot(correlation[kg], color='grey')
plt.plot(correlation[known_key_int[target_byte]], 'r')

plt.subplot(1, 2, 2)
plt.plot(key_rank_evolution_cpa, color='green')
plt.show()
