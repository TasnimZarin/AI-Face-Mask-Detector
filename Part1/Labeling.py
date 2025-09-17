import os,glob   
import pandas as pd

os.chdir(r'G:\.shortcut-targets-by-id\1noXdisFl6MYE_w2qtCawV4fozNQDCP_I\COMP-6721(AI_project)\Dataset-label')

folders = ["1", "2","3","4","5"]

files = []
count=0;
for folder in folders:
    for file in os.listdir(folder):
        if (".jpg" in file) or (".png" in file) :
            files.append([file, folder])
        elif "." not in file :
            folder2=folder+'\\'+file
            for file2 in os.listdir(folder2):
                if (".jpg" in file2) or (".png" in file2) :
                    files.append([file2, folder])
                    count=count+1;
                else:
                    continue;
            else:
                continue;

pd.DataFrame(files, columns=['image name', 'label']).to_csv('labels.csv')
