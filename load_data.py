import pandas as pd
import numpy as np

def load_data():
    data = {}

    df = pd.read_csv('/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_dorsal.csv')
    np_arr = df.to_numpy()
    data['positive'] = np.zeros((np_arr.shape[0], np_arr.shape[1], 3))
    data['positive'][:,:,0] = np_arr

    df = pd.read_csv('/store1/prj_rim/Decoding_Data/FlossieAligned/HC_dorsal.csv')
    np_arr = df.to_numpy()
    data['positive'][:,:,1] = np_arr

    df = pd.read_csv('/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_ventral.csv')
    np_arr = df.to_numpy()

    data['negative'] = np.zeros((np_arr.shape[0], np_arr.shape[1], 3))
    data['negative'][:,:,0] = np_arr

    df = pd.read_csv('/store1/prj_rim/Decoding_Data/FlossieAligned/HC_ventral.csv')
    np_arr = df.to_numpy()
    data['negative'][:,:,1] = np_arr

    return data


