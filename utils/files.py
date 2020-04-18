import os
import csv


def create_op_dir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_results(name, result_list, dirs):
    create_op_dir(dirs)
    f = open(f"{dirs}{name}.csv", "w")
    with f:
        fnames = list(result_list[0].keys())
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        for s in result_list:
            writer.writerow(s)
