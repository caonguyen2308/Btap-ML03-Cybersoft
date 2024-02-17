### BTVN
# 1. Kiểm định giả thuyết không có sự khác nhau về điểm trung bình giữa học sinh tham gia và không tham gia bài kiểm tra trước khoá học. Lựa chọn alpha = 0.05.
# 2. Hãy thực hiện theo ít nhất 1 trong những cách sau: sử dụng công thức (nếu được, tham khảo https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances_(sX1_%3E_2sX2_or_sX2_%3E_2sX1) và sử dụng thư viện.
# 3. Vẽ phân bố xác suất tương ứng và vị trí của các điểm statistic và critical.
# Cao Hoàng Khôi Nguyên - ML03

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats 
from scipy.stats import ttest_ind
from scipy.stats import t

df = pd.read_csv("C:/Users/caong/OneDrive - Thanh Thiếu Niên Đại Đạo/KN/Cybersoft Academy/Machine Learning/Thực hành/tuan02-thuchanh/Tuan02-ThucHanh/StudentsPerformance.csv")
#print (df)

# Q1
scores_columns = ['math score', 'reading score', 'writing score']
df['mean_score'] = df[scores_columns].mean(axis=1)
group1_scores = df[df['test preparation course'] == 'completed']['mean_score']
group2_scores = df[df['test preparation course'] == 'none']['mean_score']
t_statistic, p_value = ttest_ind(group1_scores, group2_scores, equal_var=False)

alpha = 0.05
if p_value < alpha:
    print("There is evidence to reject the null hypothesis: There is a difference in the mean scores between the two groups")
else:
    print("There is not enough evidence to reject the null hypothesis: There is no difference in the mean scores between the two groups")

# Q3
df = len(group1_scores) + len(group2_scores) - 2
x = np.linspace(t.pdf(0.001, df), t.ppf(0.999, df), 1000)
pdf = t.pdf(x, df)

statistic_pos = t_statistic
critical_pos = t.ppf(1-alpha/2, df)  # Critical value for two-tailed test

plt.plot(x, pdf, label='T Distribution PDF')
plt.axvline(statistic_pos, color='red', linestyle='--', label='Statistic')
plt.axvline(-statistic_pos, color='red', linestyle='--')
plt.axvline(critical_pos, color='green', linestyle='--', label='Critical Value')
plt.axvline(-critical_pos, color='green', linestyle='--')

plt.xlabel('T Value')
plt.ylabel('Probability Density')
plt.title('Probability Distribution of T-Statistic and Critical Values')
plt.legend()
plt.show()