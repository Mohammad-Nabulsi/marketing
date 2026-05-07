import pandas as pd
from pathlib import Path


def display_rules(
        rules,
        title,
        top_n=10
):
    """
    Displays rules in readable format.
    """

    print("\n" + "=" * 80)

    print(title)

    print("=" * 80)

    if rules.empty:

        print("\nNo rules found.")

        return

    for counter, (_, row) in enumerate(
            rules.head(top_n).iterrows(),
            start=1
    ):

        antecedents = (
            " + ".join(
                list(row["antecedents"])
            )
        )

        consequents = (
            " + ".join(
                list(row["consequents"])
            )
        )

        print(f"\nRULE #{counter}")

        print(
            f"\n{antecedents}"
        )

        print("  --->  ")

        print(
            f"{consequents}"
        )

        print(
            f"\nConfidence: "
            f"{row['confidence']:.3f}"
        )

        print(
            f"Lift: "
            f"{row['lift']:.3f}"
        )

        print("-" * 80)


def main():

    BASE_DIR = Path(__file__).resolve().parents[2]

    positive_rules = pd.read_pickle(
        BASE_DIR
        / "data"
        / "positive_rules.pkl"
    )

    negative_rules = pd.read_pickle(
        BASE_DIR
        / "data"
        / "negative_rules.pkl"
    )

    display_rules(
        positive_rules,
        title="TOP 10 POSITIVE RULES",
        top_n=10
    )

    display_rules(
        negative_rules,
        title="TOP 10 NEGATIVE RULES",
        top_n=10
    )


if __name__ == "__main__":
    main()