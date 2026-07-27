from s2rag.evaluation.external_adapters import BASELINE_SPECS
from s2rag.evaluation.internal_baselines import INTERNAL_BASELINES


if __name__ == "__main__":
    print("Internal baselines:")
    for name in INTERNAL_BASELINES:
        print(f"  - {name}")
    print("\nExternal baseline adapters:")
    for name, spec in BASELINE_SPECS.items():
        print(f"  - {name}: {spec.repository}")
