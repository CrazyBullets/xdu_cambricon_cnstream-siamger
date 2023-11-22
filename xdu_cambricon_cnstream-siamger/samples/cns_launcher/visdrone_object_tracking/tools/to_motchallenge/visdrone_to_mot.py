import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert gt format from Visdrone to MOT-Chanllenge')

    parser.add_argument("--filepath", dest = 'filepath', help =
                    "path of groundtrue files",
                    default = '', type = str)

    args = parser.parse_args()

    with open(args.filepath, 'r') as visdrone_gt, \
        open('gt.txt', 'w') as mot_gt:

        visdrone_gt_lines = visdrone_gt.readlines()
        for line in visdrone_gt_lines:
            frame_idx, target_id, bb_left, bb_top, bb_width, bb_height, score, _, _, _ = line.split(",")
            target_id = int(target_id) + 1
            mot_gt.write("{},{},{},{},{},{},-1,-1,-1,-1\n".format(frame_idx, target_id, bb_left, bb_top, bb_width, bb_height))



