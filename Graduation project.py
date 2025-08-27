import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import sklearn.cluster as sc
import sklearn.preprocessing as sp
#.............................................................................

sensor_data = np.array([["S1", "2025-04-28 10:00", 35.2, 12.1, 0.002],
 ["S2", "2025-04-28 10:00", 36.5, 14.0, 0.003],
["S1", "2025-04-28 11:00", 36.1, 12.5, 0.0021],
["S3", "2025-04-28 10:00", 34.0, 11.8, 0.0025],
 ["S2", "2025-04-28 11:00", 37.2, 14.3, 0.0031],
 ["S1", "2025-04-28 12:00", 37.0, 13.0, 0.0022]])

#..................................................................................
data= pd.DataFrame(sensor_data , index=[1,2,3,4,5,6],columns=["sensor_id","timestamp","temperature","stress","displacement"])
data["timestamp"]=pd.to_datetime(data["timestamp"])
data = data.set_index("sensor_id")


arr_s1= np.zeros((5,3))
arr_s2= np.zeros((5,3))
arr_s3= np.zeros((5,3))
s1=0
s2=0
s3=0
for i in sensor_data:
    if i[0]=='S1':
        arr_s1[s1][0]=i[2]
        arr_s1[s1][1]=i[3]
        arr_s1[s1][2]=i[4]
        s1+=1
    elif i[0]=='S2':
        arr_s2[s2][0] = i[2]
        arr_s2[s2][1] = i[3]
        arr_s2[s2][2] = i[4]
        s2+=1
    else :
        arr_s3[s3][0] = i[2]
        arr_s3[s3][1] = i[3]
        arr_s3[s3][2] = i[4]
        s3+=1

avg_temp_s1 = np.mean(list(map(float,(arr_s1[:,0]))))
avg_stress_s1 = np.mean(list(map(float,(arr_s1[:,1]))))
avg_dis_s1 = np.mean(list(map(float,(arr_s1[:,2]))))

avg_temp_s2 = np.mean(list(map(float,(arr_s2[:,0]))))
avg_stress_s2 = np.mean(list(map(float,(arr_s2[:,1]))))
avg_dis_s2 = np.mean(list(map(float,(arr_s2[:,2]))))

avg_temp_s3 = np.mean(list(map(float,(arr_s3[:,0]))))
avg_stress_s3 = np.mean(list(map(float,(arr_s3[:,1]))))
avg_dis_s3 = np.mean(list(map(float,(arr_s3[:,2]))))

#...........................................................................
arr_ave_stress =[avg_stress_s1,avg_stress_s2,avg_stress_s3]
max_avg_stress=np.max(arr_ave_stress)
if max_avg_stress==avg_stress_s1:
    print(f"highest average stress is sensor s1 with average = {avg_stress_s1}")
elif max_avg_stress==avg_stress_s2 :
    print(f"highest average stress is sensor s2 with average = {avg_stress_s2}")
else:
    print(f"highest average stress is sensor s3 with average = {avg_stress_s3}")

#.......................................................................................

arr_s1_extreme = sensor_data.copy()
arr_s2_extreme = sensor_data.copy()
arr_s3_extreme = sensor_data.copy()
s1=0
s2=0
s3=0
for i in sensor_data:
    if i[0]=='S1'and float(i[2])>36:
        arr_s1_extreme[s1][0] = i[0]
        arr_s1_extreme[s1][1]=i[1]
        arr_s1_extreme[s1][2]=i[2]
        arr_s1_extreme[s1][3]=i[3]
        arr_s1_extreme[s1][4]=i[4]
        s1+=1

    elif i[0]=='S2'and  float(i[2])>36:
        arr_s2_extreme[s2][0] = i[0]
        arr_s2_extreme[s2][1] = i[1]
        arr_s2_extreme[s2][2] = i[2]
        arr_s2_extreme[s2][3] = i[3]
        arr_s2_extreme[s2][4] = i[4]
        s2+=1
    elif i[0]=='S3' and  float(i[2])>36 :
        arr_s3_extreme[s3][0] = i[0]
        arr_s3_extreme[s3][1] = i[1]
        arr_s3_extreme[s3][2] = i[2]
        arr_s3_extreme[s3][3] = i[3]
        arr_s3_extreme[s3][4] = i[4]
        s3+=1
s1_new=s1
s2_new=s2
s3_new=s3
while s1_new <=5:
    arr_s1_extreme = np.delete(arr_s1_extreme , s1,axis=0)
    s1_new+=1
while s2_new <=5:
    arr_s2_extreme = np.delete(arr_s2_extreme , s2,axis=0)
    s2_new+=1
while s3_new <=5:
    arr_s3_extreme = np.delete(arr_s3_extreme , s3,axis=0)
    s3_new+=1

pt.figure('stress over time ',(20,7))
pt.plot(sensor_data[:,3],sensor_data[:,1])
pt.title('stress over time')
pt.xlabel('stress')
pt.ylabel('date')
pt.grid(True)
pt.show()

#.................................................................................

standard= sp.StandardScaler()
data_training = standard.fit_transform(sensor_data[:,[2,3,4]])

model = sc.KMeans(n_clusters=2)
result=model.fit_predict(data_training)
labels = pd.Series(result).replace({0: 'no maintenance needed ', 1: 'needed maintenance'})
data['maintenance']=labels.values
result_arr= np.array(labels)
result_arr= result_arr.reshape(6,1)
sensor_data= np.append(sensor_data,result_arr,axis=1)

