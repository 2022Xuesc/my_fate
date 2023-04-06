import torchnet.meter as tnt

losses = tnt.AverageValueMeter()
losses.add(1)
losses.add(2)
losses.add(3)
print(losses.mean)
