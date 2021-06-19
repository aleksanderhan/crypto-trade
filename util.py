import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances

from lib import delete_optuna_study, load_optuna_study, list_studies



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete_study')
    parser.add_argument('--plot_optuna')
    parser.add_argument('--list_studies', action='store_true')
    args = parser.parse_args()

    if args.delete_study:
        delete_optuna_study(args.delete_study)

    if args.plot_optuna:
        study = load_optuna_study(args.plot_optuna)

        fig1 = plot_param_importances(study)
        fig2 = plot_optimization_history(study)
        fig1.show()
        fig2.show()

    if args.list_studies:
        summaries = list_studies()
        for summary in summaries:
            print(summary.study_name)
        