"""
Main entry point for the ConSeg project.
Allows user to choose between training, inference, and preprocessing.
"""
import argparse
import sys
import train
import infer
import preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer', 'preprocess'],
                        help='Mode to run: train, infer, or preprocess')
    args = parser.parse_args()

    if args.mode == 'train':
        train.train(
            model=train.model,
            optimizer=train.optimizer,
            criterion=train.criterion,
            train_loader=train.train_loader,
            val_loader=train.val_loader,
            num_epochs=train.num_epochs,
            num_classes=train.num_classes,
            device=train.device
        )
    elif args.mode == 'infer':
        # Just run the infer.py script as is
        # (You can modularize infer.py if you want to pass arguments)
        pass  # The user can run infer.py directly for now
        print("Run infer.py directly for inference or modularize infer.py for CLI support.")
    elif args.mode == 'preprocess':
        # Just run the preprocess.py script as is
        pass  # The user can run preprocess.py directly for now
        print("Run preprocess.py directly for preprocessing or modularize preprocess.py for CLI support.")
    else:
        print("Unknown mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
