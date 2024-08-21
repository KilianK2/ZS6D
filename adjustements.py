import csv

input_file = '/home/kilian/PycharmProjects/RobotVision/ZS6D/results/results_ycbv_bop_sam_10templates_sd_dino.csv'
output_file = '/results/results_ycbv_bop_sam_10templates.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Append 0.5 to each row
        row.append('0.5')
        writer.writerow(row)

print(f"Modified CSV saved to {output_file}")