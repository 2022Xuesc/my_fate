import time

# 记录开始时间
start_time = time.perf_counter()
for i in range(10000000):
    ...
# 记录结束时间
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f'总共用时{elapsed_time}秒')