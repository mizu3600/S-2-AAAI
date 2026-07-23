from qmshe.evaluation.baselines import BASELINE_REGISTRY

if __name__ == "__main__":
    for name in BASELINE_REGISTRY:
        print(name)

