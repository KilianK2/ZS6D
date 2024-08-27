import csv

input_file = '/home/kilian/PycharmProjects/RobotVision/ZS6D/results/results1phil_ycbv-test.csv'
output_file = '/home/kilian/PycharmProjects/RobotVision/ZS6D/results/results1aphil_ycbv-test.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if row:  # Check if the row is not empty
            # Overwrite the last element with '-1'
            row[-1] = '-1'
        writer.writerow(row)

print(f"Modified CSV saved to {output_file}")