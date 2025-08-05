import argparse
from model.model import get_ratings
import pandas as pd #type: ignore

def main():
    parser = argparse.ArgumentParser(description="Run CFB QP model and prepare data for DB upload.")
    parser.add_argument('--year', type=int, required=True, help='Season year (e.g., 2024)')
    parser.add_argument('--week', type=int, default=None, help='Week number (optional, for in-season runs)')
    args = parser.parse_args()

    # Run model
    print(f"Running model for year={args.year}, week={args.week}")
    # get_ratings returns nothing, so we need to modify model.py to return ratings, records, games, fcs_losses
    results = get_ratings(args.year, args.week)
    if results is None:
        print("Model did not return results. Please check model.py to ensure it returns ratings, records, games, fcs_losses.")
        return
    ratings, slack, records = results
    print(ratings)

if __name__ == "__main__":
    main()