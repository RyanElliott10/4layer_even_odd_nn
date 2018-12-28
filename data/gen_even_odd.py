import sys
import random

def main(filename):
    f = open(filename, "w")
    for i in range(10000):
        num = random.randint(0, 100)
        polarity = num % 2
        num = '{0:08b}'.format(num)
        s = num + " " + str(polarity) + '\n'
        f.write(s)

    f.close()

if __name__ == "__main__":
    main("training_data.txt")
    main("verification_data.txt")
