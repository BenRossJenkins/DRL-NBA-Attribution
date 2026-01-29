"""
NBA Data Loader using nba-on-court

This script uses the nba_on_court library to:
1. Download play-by-play data from the nba_data repository
2. Add on-court player information for each event
3. Optionally replace player IDs with names
4. Save processed data in the correct format for RAPM calculations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import List, Optional, Union

try:
    import nba_on_court as noc
except ImportError:
    print("Error: nba_on_court library not found. Please install it with:")
    print("  pip install nba-on-court")
    sys.exit(1)


def load_nba_data(
    seasons: Union[int, List[int], range],
    data: Union[str, tuple] = 'nbastats',
    seasontype: str = 'regular',
    untar: bool = True,
    path: Optional[str] = None
) -> List[str]:
    """
    Download NBA play-by-play data using nba_on_court.
    
    Args:
        seasons: Season year(s) to download (e.g., 2022 or range(2018, 2023))
        data: Data source(s) - 'nbastats', 'pbpstats', 'data nba', 'shot detail', or tuple
        seasontype: 'regular' or 'po' (playoffs)
        untar: Whether to unzip files immediately
        path: Directory to save files (default: current directory)
    
    Returns:
        List of downloaded file paths
    """
    print(f"Downloading NBA data for season(s): {seasons}")
    print(f"Data source: {data}, Season type: {seasontype}")
    
    # Set default path to current directory if None
    if path is None:
        base_path = Path.cwd()
        download_path = str(base_path)
    else:
        base_path = Path(path)
        download_path = path
    
    try:
        noc.load_nba_data(
            seasons=seasons,
            data=data,
            seasontype=seasontype,
            untar=untar,
            path=download_path
        )
        
        # Determine file extension
        ext = '.csv' if untar else '.tar.xz'
        
        # Find files based on season and data type
        if isinstance(seasons, (int, np.integer)):
            seasons = [seasons]
        elif isinstance(seasons, range):
            seasons = list(seasons)
        
        files = []
        data_types = data if isinstance(data, tuple) else [data]
        
        for season in seasons:
            for data_type in data_types:
                pattern = f"{data_type}_{season}{ext}"
                found_files = list(base_path.glob(pattern))
                files.extend(found_files)
        
        print(f"Downloaded {len(files)} file(s)")
        return [str(f) for f in files]
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def load_and_combine_csvs(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.
    
    Args:
        file_paths: List of CSV file paths
    
    Returns:
        Combined DataFrame
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    
    print(f"Loading {len(file_paths)} CSV file(s)...")
    dataframes = []
    
    for file_path in file_paths:
        print(f"  Loading {Path(file_path).name}...")
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    print(f"Combined data: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
    
    return combined_df


def add_players_on_court(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add on-court player information using nba_on_court.
    
    This adds 10 columns: AWAY_PLAYER1-5 and HOME_PLAYER1-5
    
    Processes games one at a time to handle errors gracefully and ensure
    proper index handling.
    
    Args:
        df: Play-by-play DataFrame
    
    Returns:
        DataFrame with added player columns
    """
    if df.empty:
        print("Warning: Empty DataFrame, skipping players_on_court")
        return df
    
    print("Adding on-court player information...")
    print(f"  Input shape: {df.shape}")
    print(f"  Processing {df['GAME_ID'].nunique()} games...")
    
    # Process game by game to avoid index issues and handle errors gracefully
    results = []
    errors = []
    game_ids = df['GAME_ID'].unique()
    
    try:
        from tqdm import tqdm
        progress_bar = tqdm(game_ids, desc="  Processing games")
    except ImportError:
        progress_bar = game_ids
    
    for game_id in progress_bar:
        game_df = df[df['GAME_ID'] == game_id].copy()
        
        # Reset index to ensure it starts at 0 (required by nba_on_court)
        game_df = game_df.reset_index(drop=True)
        
        try:
            game_result = noc.players_on_court(game_df)
            results.append(game_result)
        except Exception as e:
            errors.append((game_id, str(e)))
            # Add empty lineup columns to maintain structure
            for i in range(1, 6):
                game_df[f'AWAY_PLAYER{i}'] = None
                game_df[f'HOME_PLAYER{i}'] = None
            results.append(game_df)
    
    if errors:
        print(f"  Warning: {len(errors)} games had errors (added empty lineup columns)")
        if len(errors) <= 10:
            for game_id, error in errors:
                print(f"    Game {game_id}: {error}")
        else:
            print(f"    First 5 errors:")
            for game_id, error in errors[:5]:
                print(f"      Game {game_id}: {error}")
    
    # Combine all results
    if results:
        df_with_lineups = pd.concat(results, axis=0, ignore_index=True)
        
        # Check which columns were added
        new_cols = [col for col in df_with_lineups.columns if col not in df.columns]
        if new_cols:
            print(f"  Added columns: {new_cols}")
        print(f"  Output shape: {df_with_lineups.shape}")
        
        return df_with_lineups
    else:
        print("  No games processed successfully")
        return df


def replace_player_ids_with_names(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Replace player IDs with player names in specified columns.
    
    Args:
        df: DataFrame with player ID columns
        columns: List of column names to replace (default: all AWAY_PLAYER and HOME_PLAYER columns)
    
    Returns:
        DataFrame with player names instead of IDs
    """
    if columns is None:
        # Find all AWAY_PLAYER and HOME_PLAYER columns
        columns = [col for col in df.columns if col.startswith('AWAY_PLAYER') or col.startswith('HOME_PLAYER')]
    
    if not columns:
        print("No player columns found to replace")
        return df
    
    print(f"Replacing player IDs with names in {len(columns)} columns...")
    
    try:
        df[columns] = df[columns].apply(noc.players_name, result_type="expand")
        print("Player names replaced successfully")
        return df
    except Exception as e:
        print(f"Error replacing player names: {e}")
        print("Continuing with player IDs...")
        return df


def process_data(
    seasons: Union[int, List[int], range],
    data: Union[str, tuple] = 'nbastats',
    seasontype: str = 'regular',
    add_lineups: bool = True,
    add_names: bool = False,
    output_file: Optional[str] = None,
    output_format: str = 'parquet',
    cleanup: bool = False
) -> pd.DataFrame:
    """
    Complete pipeline: download, load, process, and save NBA data.
    
    Args:
        seasons: Season year(s) to process
        data: Data source(s)
        seasontype: 'regular' or 'po'
        add_lineups: Whether to add on-court player information
        add_names: Whether to replace player IDs with names
        output_file: Output file path (auto-generated if None)
        output_format: 'parquet' or 'csv'
        cleanup: Whether to delete downloaded CSV files after processing
    
    Returns:
        Processed DataFrame
    """
    # Step 1: Download data
    print("\n" + "="*60)
    print("STEP 1: Downloading data")
    print("="*60)
    file_paths = load_nba_data(seasons, data, seasontype, untar=True)
    
    if not file_paths:
        raise ValueError("No files were downloaded")
    
    # Step 2: Load and combine CSVs
    print("\n" + "="*60)
    print("STEP 2: Loading and combining data")
    print("="*60)
    df = load_and_combine_csvs(file_paths)
    
    # Step 3: Add on-court players
    if add_lineups:
        print("\n" + "="*60)
        print("STEP 3: Adding on-court player information")
        print("="*60)
        df = add_players_on_court(df)
    
    # Step 4: Replace IDs with names (optional)
    if add_names:
        print("\n" + "="*60)
        print("STEP 4: Replacing player IDs with names")
        print("="*60)
        df = replace_player_ids_with_names(df)
    
    # Step 5: Save processed data
    print("\n" + "="*60)
    print("STEP 5: Saving processed data")
    print("="*60)
    
    if output_file is None:
        # Generate output filename
        if isinstance(seasons, (int, np.integer)):
            season_str = str(seasons)
        elif isinstance(seasons, range):
            season_str = f"{seasons.start}_{seasons.stop-1}"
        else:
            season_str = f"{min(seasons)}_{max(seasons)}"
        
        data_str = '_'.join(data) if isinstance(data, tuple) else data
        output_file = f"nba_pbp_{season_str}_{data_str}_{seasontype}.{output_format}"
    
    output_path = Path(output_file)
    
    if output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Saved processed data to: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Step 6: Cleanup (optional)
    if cleanup:
        print("\n" + "="*60)
        print("STEP 6: Cleaning up downloaded files")
        print("="*60)
        for file_path in file_paths:
            try:
                Path(file_path).unlink()
                print(f"  Deleted: {Path(file_path).name}")
            except Exception as e:
                print(f"  Could not delete {Path(file_path).name}: {e}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Load and process NBA play-by-play data using nba-on-court',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process 2022-23 regular season
  python data_loader.py --seasons 2022
  
  # Download multiple seasons
  python data_loader.py --seasons 2018 2019 2020 2021 2022
  
  # Download playoffs data
  python data_loader.py --seasons 2022 --seasontype po
  
  # Add player names instead of IDs
  python data_loader.py --seasons 2022 --add-names
  
  # Save as CSV instead of Parquet
  python data_loader.py --seasons 2022 --format csv
        """
    )
    
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        required=True,
        help='Season year(s) to download (e.g., 2022 or 2018 2019 2020)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='nbastats',
        choices=['nbastats', 'pbpstats', 'data nba', 'shot detail'],
        help='Data source (default: nbastats)'
    )
    parser.add_argument(
        '--seasontype',
        type=str,
        default='regular',
        choices=['regular', 'po'],
        help='Season type: regular or po (playoffs) (default: regular)'
    )
    parser.add_argument(
        '--add-lineups',
        action='store_true',
        default=True,
        help='Add on-court player information (default: True)'
    )
    parser.add_argument(
        '--no-lineups',
        dest='add_lineups',
        action='store_false',
        help='Skip adding on-court player information'
    )
    parser.add_argument(
        '--add-names',
        action='store_true',
        help='Replace player IDs with names'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete downloaded CSV files after processing'
    )
    
    args = parser.parse_args()
    
    # Convert single season to list if needed
    seasons = args.seasons[0] if len(args.seasons) == 1 else args.seasons
    
    try:
        df = process_data(
            seasons=seasons,
            data=args.data,
            seasontype=args.seasontype,
            add_lineups=args.add_lineups,
            add_names=args.add_names,
            output_file=args.output,
            output_format=args.format,
            cleanup=args.cleanup
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Processed {len(df):,} events")
        if args.add_lineups:
            player_cols = [col for col in df.columns if 'PLAYER' in col and col.startswith(('AWAY_', 'HOME_'))]
            if player_cols:
                print(f"Added {len(player_cols)} on-court player columns")
                print(f"Sample columns: {player_cols[:5]}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
