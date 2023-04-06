import csv
import time

file = open('test.csv', 'w', buffering=1)
writer = csv.writer(file)
for i in range(10000):
    writer.writerow(['hello', 'world'])
    time.sleep(1)
