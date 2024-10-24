#%%
# Import Dependencies
from matplotlib.pyplot import figure
from pygeom.tools.spacing import (equal_spacing, full_cosine_spacing,
                                  linear_bias_left, linear_bias_right,
                                  semi_cosine_spacing)

#%%
# Create Spacings
num = 10

s1 = equal_spacing(num)
s2 = semi_cosine_spacing(num)

print(f'equal = {s1}\n')
print(f'semi-cosine = {s2}\n')

num = 20

s3 = equal_spacing(num)
s4 = full_cosine_spacing(num)

print(f'equal = {s3}\n')
print(f'full-cosine = {s4}\n')

#%%
# Bias Equal Spacing
ratio = 0.2

s = s3

s5 = linear_bias_left(s, ratio)
s6 = linear_bias_right(s, ratio)

print(f'linear bias left = {s5}\n')
print(f'linear bias right = {s6}\n')

n = int((len(s)-1)/2)

print(f's5[n] = {s5[n]}')
print(f's6[n] = {s6[n]}')

#%%
# Plots
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
p1 = ax.plot(s1, label='s1')
p2 = ax.plot(s2, label='s2')
p3 = ax.plot(s3, label='s3')
p4 = ax.plot(s4, label='s4')
p5 = ax.plot(s5, label='s5')
p6 = ax.plot(s6, label='s6')
l = ax.legend()
