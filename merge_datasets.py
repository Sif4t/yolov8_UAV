"""
Merge any number of YOLO-format (Roboflow-exported) detection datasets into one combined dataset,
building a unified class list even when the input datasets have different classes.

Usage:
    python3 merge_datasets.py --datasets DATASET_DIR_1 DATASET_DIR_2 ... --out MERGED_DIR

Each DATASET_DIR is expected to look like a standard Roboflow YOLOv8 export:
    DATASET_DIR/
      data.yaml        # contains a `names:` list
      train/images, train/labels
      valid/images, valid/labels
      test/images,  test/labels   (optional)
"""
import argparse
import os
import shutil
import yaml


def load_class_names(dataset_dir):
    with open(os.path.join(dataset_dir, "data.yaml"), "r") as f:
        data = yaml.safe_load(f)
    return list(data["names"])


def build_global_classes(dataset_dirs):
    global_classes = []
    per_dataset_names = []
    for d in dataset_dirs:
        names = load_class_names(d)
        per_dataset_names.append(names)
        for n in names:
            if n not in global_classes:
                global_classes.append(n)
    return global_classes, per_dataset_names


def remap_and_copy_split(dataset_dir, split, local_names, global_classes, out_dir, tag):
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")
    if not os.path.isdir(img_dir):
        return 0

    out_img_dir = os.path.join(out_dir, split, "images")
    out_lbl_dir = os.path.join(out_dir, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    remap = {i: global_classes.index(name) for i, name in enumerate(local_names)}
    count = 0

    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        new_stem = f"{tag}_{stem}"

        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_stem + ext))

        src_lbl = os.path.join(lbl_dir, stem + ".txt")
        dst_lbl = os.path.join(out_lbl_dir, new_stem + ".txt")
        if os.path.exists(src_lbl):
            with open(src_lbl, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                parts[0] = str(remap[old_cls])
                new_lines.append(" ".join(parts))
            with open(dst_lbl, "w") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        else:
            # negative example (no objects) — still valid, write empty label file
            open(dst_lbl, "w").close()

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    global_classes, per_dataset_names = build_global_classes(args.datasets)
    print(f"Merged class list ({len(global_classes)}): {global_classes}")

    for dataset_dir, local_names in zip(args.datasets, per_dataset_names):
        tag = os.path.basename(os.path.normpath(dataset_dir))
        for split in ("train", "valid", "test"):
            n = remap_and_copy_split(dataset_dir, split, local_names, global_classes, args.out, tag)
            if n:
                print(f"  {dataset_dir} [{split}]: {n} images -> merged/{split}")

    merged_yaml = {
        "path": os.path.abspath(args.out),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(global_classes),
        "names": global_classes,
    }
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump(merged_yaml, f, sort_keys=False)

    print(f"\nWrote merged dataset to {args.out}/data.yaml")


if __name__ == "__main__":
    main()
