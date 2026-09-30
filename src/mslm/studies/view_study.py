import optuna
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Cargar un estudio existente de Optuna")
    parser.add_argument("--study_name", type=str, required=True, help="Nombre del estudio")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    study_name = args.study_name
    storage = f"sqlite:///{study_name}.db"  # Ej: "sqlite:///optuna_study.db"

    # Carga el study existente
    study = optuna.load_study(study_name=study_name, storage=storage)

    for i, trial in enumerate(study.trials):
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print(f"  Params: {trial.params}")
        print(f"  State: {trial.state}")
        print(f"  User attrs: {trial.user_attrs}")
        print(f"  System attrs: {trial.system_attrs}")
        print()
    
    # Imprime los resultados
    print("Resultados del estudio:")
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)