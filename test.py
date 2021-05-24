import requests
import pandas as pd





def get_data(frame_size, start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?frame_size={frame_size}&start_time={start_time}&end_time={end_time}') # Use frame_size to get frame_size amount of datapoints
    
    #print(r.json())

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df




frame_size = 10
start_time = '2021-05-22T00:00'
end_time = '2021-05-24T00:00'


get_data(frame_size, start_time, end_time)