import EE
import EEDataSetGenerator

num = int(input("last file/gen number: "))
while True:
    EEDataSetGenerator.Main(num)
    num += 1
    EE.Main(num)
    
