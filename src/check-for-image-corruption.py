import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sample_info = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/KF_5_in_5_tfrecords.tsv',sep='\t')
df1, df2, df3, df4, df5 = np.array_split(sample_info,5)

data = df5.copy()

size = len(data['Tile'])
i=0
for path in data['Tile']:
    try:
        img = plt.imread(path)
    except:
        print(path + ' is problematic')
    if i % 10000 == 0:
        print('Tile '+ str(i) + '/'+ str(size))
    i+=1
