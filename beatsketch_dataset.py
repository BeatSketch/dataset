#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import colorama
import multiprocessing as mp
import util.cli as cli

if __name__ == "__main__":
    args, ap = cli.parse_args()
    try:
        from util.manager import folder_preprocessing, process_file, process_folder
        import ml
        from util import loader_exporter
    except ModuleNotFoundError as e:
        print(
            colorama.Style.DIM
            + colorama.Fore.RED
            + "Not all required python modules are installed."
        )
        print(
            colorama.Fore.RESET
            + "To fix, run `python -m pip install -r requirements.txt` in this folder"
        )
        if input("Print stack trace? (y/N) ").lower() == "y":
            raise e
        exit(1)

    if args.cmd == "preprocess":
        if args.preprocess == "file":
            process_file(
                args.file,
                False,
                processed_save_location=args.save_path,
                no_block_ratio=args.split,
            )
        elif args.preprocess == "folder":
            process_folder(
                args.folder,
                False,
                processed_save_location=args.save_path,
                max_files=mp.cpu_count() if args.test else -1,
                no_block_ratio=args.split,
            )
        elif args.preprocess == "bpm":
            folder_preprocessing(args.folder)
    elif args.cmd == "train":
        if args.train == "file":
            process_file(
                args.file,
                True,
                processed_save_location=args.save_path,
                test_onnx=args.onnx,
                model=args.model,
                no_block_ratio=args.split,
            )
        elif args.train == "folder":
            process_folder(
                args.folder,
                True,
                processed_save_location=args.save_path,
                max_files=mp.cpu_count() if args.test else -1,
                test_onnx=args.onnx,
                model=args.model,
                no_block_ratio=args.split,
            )
        elif args.train == "dataset":
            dataset = loader_exporter.import_dataset(args.dataset)
            print("\n==> Starting training")
            ml.train_with_existing_dataset(
                dataset, test_onnx=args.onnx, model=args.model
            )
    elif args.cmd == "eval":
        print("\n  MODEL EVALUATION\n")
        ml.test_model(loader_exporter.import_dataset(args.dataset), args.model_path)
    elif args.cmd == "help":
        ap.print_help()
