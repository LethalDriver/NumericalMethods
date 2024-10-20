import math
import time

start_time = time.time()
k = 1
sum = 0

while True:
    n = 1/k**2
    prev = sum
    sum += n
    if sum == prev:
        break
    k += 1

analystical_value = math.pi**2/6
percentage_difference = (1-(sum/analystical_value))*100
execution_time = time.time() - start_time

print(f"Sum for n max: {sum:.10f}")
print(f"Relatvie difference from analytical value (percentage): {percentage_difference:.10f}%")
print(f"Execution time: {execution_time:.10f} seconds")

