import pandas as pd
import numpy as np
# df1 = pd.read_csv('csv/Briareo/normal.csv', header = None) # place your csv1 in df1

#df1[df1.columns.drop('A')]
o = pd.read_csv('csv/Briareo/original.csv', header = None) 
df1 = pd.read_csv('csv/Briareo/normal.csv', header = None) # place your csv1 in df1
df2 = pd.read_csv('csv/Briareo/depth.csv', header = None) # place your csv2 in df2
df3 = pd.read_csv('csv/Briareo/ir.csv', header = None) # place your csv2 in df2
df4 = pd.read_csv('csv/Briareo/rgbop.csv', header = None) # place your csv2 in df2
df5 = pd.read_csv('csv/Briareo/rgb.csv', header = None) # place your csv2 in df2
#df2[df2.columns.drop('A')]
#df4 = pd.read_csv('csv/Briareo/color.csv', header = None) 

o1 = o.iloc[:,:].values.tolist() 
#print(type(o1))

rate_in_1 = df1.iloc[:,:].values.tolist() #store the values of the 3rd column from csv1 to a list
rate_out_1 = df2.iloc[:,:].values.tolist() #store the values of the 4th column from csv1 to a list
rate_out_2 = df3.iloc[:,:].values.tolist() #store the values of the 4th column from csv1 to a list
rate_in_2 = df4.iloc[:,:].values.tolist() #store the values of the 4th column from csv1 to a list
rate_in_5 = df5.iloc[:,:].values.tolist() #store the values of the 4th column from csv1 to a list


# rate_in_2 = df2.iloc[:,2].values.tolist() #store the values of the 3rd column from csv1 to a list
#rate_out_2 = df2.iloc[:,3].values.tolist() #store the values of the 4th column from csv1 to a list

 # add the values of 2 rate in lists into rate_in_total list
# rate_in_total = [x+y for x, y in zip(rate_in_1, rate_out_1)] # add the values of 2 rate out lists into rate_out_total list
# rate_in_total = [max(x,y) for (x, y) in zip(rate_in_1, rate_out_2)]
# rate_in_total = [np.add(x,y)/2 for (x, y) in zip(rate_in_1, rate_out_1)]
# rate_in_total = [max(max(x,y),z) for (x, y, z) in zip(rate_in_1, rate_out_1, rate_out_2)]
# rate_in_total = [np.add(np.add(x,y),z)/3 for (x, y, z) in zip(rate_in_1, rate_out_1, rate_out_2)]
# rate_in_total = [np.add(np.add(np.add(x,y),w),z)/4 for (x, y,z,w) in zip(rate_in_1, rate_out_1,rate_out_2,rate_in_2)]
rate_in_total = [np.add(np.add(np.add(np.add(x,y),w),z),k)/5 for (x, y,z,w,k) in zip(rate_in_1, rate_out_1,rate_out_2,rate_in_2,rate_in_5)]
#print(rate_in_total[1]) 

final_df = pd.DataFrame(rate_in_total)
#print(final_df)
with open('csv/Briareo/ir_rgb.csv', 'a', newline='') as csvfile:
	final_df.to_csv(csvfile, mode='a',header=False,index =False)
	# print(csvfile)
#print(np.where(max(rate_in_total[1])))

#print(len(rate_in_total))
c=0
for x in range(len(rate_in_total)):
	#print(np.argmax(rate_in_total[x], axis=0))
	#print(o1[x])
	#print(np.argmax(rate_in_total[x], axis=0)==o1[x])
	if np.argmax(rate_in_total[x], axis=0)==o1[x]:
		c +=1
#print(c)		
print( c / 218)
# print( c / 482)
