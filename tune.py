import json
import subprocess
from pathlib import Path
import optuna
from train_ppo_predators import train 
SUBMISSION_DIR = Path("submissions/mayamahouachi/tuna")
RESULT_JSON = Path("results/latest_evaluation.json")
MODEL_PATH = SUBMISSION_DIR / "shared_predator_model.pth"
BEST_PATH = SUBMISSION_DIR / "best_shared_predator_model.pth"
DB = "sqlite:///optuna_ppo_simple_tag.db"
STUDY_NAME = "ppo_simple_tag"
def eval_score(episodes: int = 50):
    subprocess.run(["python", "evaluate.py", "--submission-dir", str(SUBMISSION_DIR), "--episodes", str(episodes)],check=True)
    data = json.loads(RESULT_JSON.read_text())
    return float(data["predator_score"])

def objective(trial: optuna.Trial):
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 5e-4, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    epsilon_clip = trial.suggest_float("epsilon_clip", 0.10, 0.30)
    K_epochs = trial.suggest_int("K_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.03)
    batch_size = trial.suggest_categorical("batch_size", [10, 20, 30])
    train(num_episodes=600,max_cycles=200,batch_size=batch_size,save_path=str(MODEL_PATH),lr_actor=lr_actor,lr_critic=lr_critic,gamma=gamma,epsilon_clip=epsilon_clip,
              K_epochs=K_epochs,gae_lambda=gae_lambda,entropy_coef=entropy_coef)
    score = eval_score(episodes=500)
    trial_model_path = SUBMISSION_DIR / f"trial_{trial.number:04d}_model.pth"
    trial_model_path.write_bytes(MODEL_PATH.read_bytes())
    trial.set_user_attr("model_path", str(trial_model_path))
    return score

def on_trial_complete(study: optuna.Study, trial: optuna.trial.FrozenTrial) :
    if study.best_trial.number == trial.number:
        mp = trial.user_attrs.get("model_path", None)
        if mp is not None:
            BEST_PATH.write_bytes(Path(mp).read_bytes())
            print(f"Update ; score={trial.value:.2f} for {BEST_PATH}")

if __name__ == "__main__":
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="maximize",study_name=STUDY_NAME,storage=DB,load_if_exists=True)
    study.optimize(objective, n_trials=20, callbacks=[on_trial_complete])
    print("Best score :", study.best_value)
    print("Best params:", study.best_params)
    print("Best model:", BEST_PATH)