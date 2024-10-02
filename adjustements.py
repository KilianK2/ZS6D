import csv

input_file = '/home/kilian/PycharmProjects/RobotVision/ZS6D/results/results4_ycbv-test.csv'
output_file = '/home/kilian/PycharmProjects/RobotVision/ZS6D/results/results4a_ycbv-test.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if row:  # Check if the row is not empty
            # Append '-1' to the end of the row
            row.append('-1')
        writer.writerow(row)

print(f"Modified CSV saved to {output_file}")