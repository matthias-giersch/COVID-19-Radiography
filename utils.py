import argparse
import os
import random
from pathlib import Path
import shutil


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_dataset(
    base_dir: Path,
    output_base: Path = Path("dataset_split"), 
    train_ratio: float=0.7, 
    valid_ratio: float=0.15, 
    test_ratio: float=0.15, 
    seed=42
) -> None:
    random.seed(seed)

    class_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nClass: {class_name}")

        train_dir = output_base / "train" / class_name
        valid_dir = output_base / "valid" / class_name
        test_dir  = output_base / "test" / class_name
        for d in [train_dir, valid_dir, test_dir]:
            make_dir(d)

        images = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_valid = int(valid_ratio * n_total)
        n_test = n_total - n_train - n_valid

        train_files = images[:n_train]
        valid_files = images[n_train:n_train + n_valid]
        test_files  = images[n_train + n_valid:]

        print(f"→ {n_total} images: train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")

        for fpath in train_files:
            shutil.copy(fpath, train_dir / fpath.name)
        for fpath in valid_files:
            shutil.copy(fpath, valid_dir / fpath.name)
        for fpath in test_files:
            shutil.copy(fpath, test_dir / fpath.name)

    print("\nData split done!")
    print(f"Results at: {os.path.abspath(output_base)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images in train/valid/test.")
    parser.add_argument("--input_path", help="Path to folder with images")
    parser.add_argument("--output", default="images", help="Name of putput folder")
    parser.add_argument("--train", type=float, default=0.7, help="Percentage of trainings data")
    parser.add_argument("--valid", type=float, default=0.15, help="Percentage of validation data")
    parser.add_argument("--test",  type=float, default=0.15, help="Percentage of test data")
    args = parser.parse_args()

    split_dataset(Path(args.input_path), Path(args.output), args.train, args.valid, args.test)
