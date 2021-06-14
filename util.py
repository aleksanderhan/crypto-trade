import argparse

from lib import delete_optuna_study



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete_study')
    args = parser.parse_args()

    if args.delete_study:
        delete_optuna_study(args.delete_study)