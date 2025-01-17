csv_name = 'yt-vad-650-clean-train.csv'
import pandas as pd
import subprocess
pwd = subprocess.check_output(['pwd']).decode('utf-8')
pwd = pwd[:-1]
df = pd.read_csv(csv_name)
df = df.drop('wav_filesize', axis = 1)
df['wav_filename'] = df['wav_filename'].apply(lambda x: pwd+x[26:])
for index, row in df.iterrows():
    content = open(row['wav_filename'][:-4]+'.txt', "wb")
    content.write(row['transcript'].encode('utf-8'))
    row['transcript'] = row['wav_filename'][:-4]+'.txt'
df.to_csv(csv_name, index=False, header=False)
