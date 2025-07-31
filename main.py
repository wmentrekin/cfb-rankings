import argparse
from model.model import get_ratings
from database.model_to_df import ratings_to_df, games_to_df, fcs_losses_to_df
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run CFB QP model and prepare data for DB upload.")
    parser.add_argument('--year', type=int, required=True, help='Season year (e.g., 2024)')
    parser.add_argument('--week', type=int, default=None, help='Week number (optional, for in-season runs)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output CSVs')
    args = parser.parse_args()

    # Run model
    print(f"Running model for year={args.year}, week={args.week}")
    # get_ratings returns nothing, so we need to modify model.py to return ratings, records, games, fcs_losses
    results = get_ratings(args.year, args.week)
    if results is None:
        print("Model did not return results. Please check model.py to ensure it returns ratings, records, games, fcs_losses.")
        return
    ratings, records, games, fcs_losses = results

    # Convert to DataFrames
    df_ratings = ratings_to_df(ratings, records)
    df_games = games_to_df(games)
    df_fcs = fcs_losses_to_df(fcs_losses)

    # Save to CSVs for now (future: upload to DB)
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    df_ratings.to_csv(f"{args.output_dir}/ratings.csv", index=False)
    df_games.to_csv(f"{args.output_dir}/games.csv", index=False)
    df_fcs.to_csv(f"{args.output_dir}/fcs_losses.csv", index=False)
    print(f"Output written to {args.output_dir}/ratings.csv, games.csv, fcs_losses.csv")

if __name__ == "__main__":
    main()