import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('datasets/titanic.csv')
num_df = df.select_dtypes('number')
num_df.fillna(num_df.mean(),inplace=True)
# num_df.dropna(inplace=True)
print(num_df)


# plt.plot and plt.scatter from Matplotlib both visualize data points, but differ in functionality and customization

# PLOT
# plt.plot: Primarily for line plots, connecting data points with lines. It can also create basic scatter plots, but with limited customization options for individual points. It is faster for simple scatter plots with many points.

# plt.plot(num_df['Age'],num_df['Fare'],'o')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Plot')
# plt.show()

# SCATTER
# plt.scatter: Specifically designed for scatter plots. It allows extensive customization of individual data points (size, color, shape). It is suitable for complex visualizations where point properties are mapped to data attributes.

# plt.scatter(num_df['Age'],num_df['Fare'])
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Scatter')
# plt.show()


# PIE

# not_survived = (num_df['Survived'] == 0).sum()  
# survived = (num_df['Survived'] == 1).sum()
# print(f"ns = {not_survived}")
# print(f"ns = {survived}")
# age_group = np.array([not_survived,survived])
# plt.pie(age_group,labels=[f'Not Survived {not_survived}',f'Survived {survived}'])
# plt.show()

# HISTOGRAM

# plt.hist(num_df['Age'])
# plt.xlabel('Age')
# plt.ylabel('No. of Passengers')
# plt.show()