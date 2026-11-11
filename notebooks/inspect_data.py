"""Quick profile of the raw credit-risk dataset.

Run from the project root:

    python notebooks/inspect_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import load_raw_data  # noqa: E402


def main() -> None:
    df = load_raw_data()

    print("أول 5 أسطر من البيانات:")
    print(df.head())

    print("\nحجم البيانات (الصفوف، الأعمدة):")
    print(df.shape)

    print("\nمعلومات الأعمدة وأنواع البيانات:")
    df.info()

    print("\nعدد القيم المفقودة في كل عمود:")
    print(df.isnull().sum())

    print("\nملخص إحصائي للأعمدة الرقمية:")
    print(df.describe())


if __name__ == "__main__":
    main()
