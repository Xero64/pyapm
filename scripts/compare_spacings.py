#%% Import Dependencies
from pyapm.tools import equal_spacing, semi_cosine_spacing, full_cosine_spacing

#%% Create Spacings
num = 10

s1 = equal_spacing(num)
s2 = semi_cosine_spacing(num)

print(f'equal = \n{s1}\n')
print(f'semi-cosine = \n{s2}\n')

num = 20

s3 = equal_spacing(num)
s4 = full_cosine_spacing(num)

print(f'equal = \n{s3}\n')
print(f'full-cosine = \n{s4}\n')
