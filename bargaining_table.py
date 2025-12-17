
# List all the files in a folder
import os
import random
from typing import List, Tuple

def split_into_blocks(csv_path):
    """Splits a CSV file into blocks based on game numbers.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        List[List[str]]: A list of blocks, where each block is a list of strings (lines). Block 0 is the header.
    """
    blocks = []
    current_block = []

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip newline but keep commas structure
            stripped = line.rstrip("\n")
            if not stripped.strip(): 
                # Skip completely empty lines
                continue

            # Try to detect the start of a new block
            parts = stripped.split(",")
            first_cell = parts[0].strip()

            # A block starts when first cell is an integer game number (1,2,...)
            if first_cell.isdigit():
                # If there is an existing block, save it
                if current_block:
                    blocks.append(current_block)
                current_block = [stripped]  # Start new block
            else:
                # Inside a block → append line
                if current_block is not None:
                    current_block.append(stripped)

        # Append last block if exists
        if current_block:
            blocks.append(current_block)

    return blocks
        
def parse_line(line: str) -> Tuple[int, str]:
    """Parses a line from the CSV file.

    Args:
        line (str): A line from the CSV file.

    Returns:
        Tuple[int, str]: A tuple containing the value and assignment.
    """
    parts = line.split(",")
    value = int(parts[9].strip())
    assignment = parts[10].strip()
    return value, assignment

def compute_value(assignments_player1:List[Tuple[int, str]], assignments_player2:List[Tuple[int, str]]) -> Tuple[float, float]:
    """Computes the values for both players based on their assignments.

    Args:
        assignments_player1 (List[Tuple[int, str]]): The assignments for player 1.
        assignments_player2 (List[Tuple[int, str]]): The assignments for player 2.

    Returns:
        Tuple[float, float]: The computed values for both players.
    """
    assert len(assignments_player1) == len(assignments_player2)
    v_player1, v_player2 = 0.0, 0.0
    for a1, a2 in zip(assignments_player1, assignments_player2):
        if a1[1] != a2[1]:
            v_player1 -= 0.2 * a1[0]
            v_player2 -= 0.2 * a1[0]
        elif a1[1] == a2[1] and a1[1] == "r1":
            v_player1 += a1[0]
        elif a1[1] == a2[1] and a1[1] == "r2":
            v_player2 += a1[0]

    return (v_player1, v_player2)

def game_signature(csv_file: str) -> str:
    """ 
    Generates a signature for the game based on the CSV file.
    It can be used to check whether two games can be compared.
    It extracts the X,Y position of each disk and interaction and concatenate them.
    Args:
        csv_file (str): The path to the CSV file.
    Returns:
        str: The game signature.
    """
    blocks = split_into_blocks(csv_file)[1:]  # Skip header block
    signature = ""
    for b in blocks:
        for row in b[3:]:  # Skip first 3 rows of each block
            row_split = row.split(',')
            x, y = row_split[-4], row_split[-3]  # XY positions in the original string
            signature += f"{x},{y};"
    return signature

def player_files(folder_path: str, file_prefix: str = "") -> List[str]:
    files = os.listdir(folder_path)
    player_files = []
    for f in files:
        if file_prefix in f:
            player_files.append(f)
    return player_files

def sample_pairs(
    player1_files: List[str], 
    player2_files: List[str], 
    num_samples: int, 
    with_replacement: bool = False
    ) -> List[Tuple[str, str]]:
    assert len(player1_files) > 0 and len(player2_files) > 0, f"Cannot sample pairs! len(player1_files)={len(player1_files)}, len(player2_files)={len(player2_files)}"
    sampled_pairs = []
    available_pairs = [(f1, f2) for f1 in player1_files for f2 in player2_files]
    
    if with_replacement:
        for _ in range(num_samples):
            sampled_pairs.append(random.choice(available_pairs))
    else:
        assert len(player1_files)*len(player2_files) >= num_samples, f"Cannot sample more pairs than available ({len(player1_files)*len(player2_files)}) without replacement!"
        sampled_pairs = random.sample(available_pairs, min(num_samples, len(available_pairs)))
    
    return sampled_pairs

if __name__ == "__main__":
    # Global parameters
    game_num = 2 # The game we want to analyse
    num_samples = 100 # Number of random pairs to sample
    sample_with_replacement = False # Whether to sample with replacement
    folder_path_player1 = "./data/Dor-humans/stage-2-analysis/Number1players/"
    folder_path_player2 = "./data/Dor-humans/stage-2-analysis/Number2players/"

    assert 1 <= game_num <= 10, "game_num must be between 1 and 10"

    # Retrieve all the games played by both players
    player1_files = player_files(folder_path_player1, "BT")
    player2_files = player_files(folder_path_player2, "BT")

    sampled_pairs = sample_pairs(
        player1_files,
        player2_files,
        num_samples,
        with_replacement=sample_with_replacement
    )

    for game_num in range(1,11):
        count = 0
        player1_payoff, player2_payoff = 0.0, 0.0
        for f1, f2 in sampled_pairs:
            sign_f1 = game_signature(os.path.join(folder_path_player1, f1))
            sign_f2 = game_signature(os.path.join(folder_path_player2, f2))
            if sign_f1 != sign_f2:
                raise ValueError(f"Game signatures do not match for files {f1} and {f2}!")
            
            blocks1 = split_into_blocks(os.path.join(folder_path_player1, f1))[game_num][3:]
            blocks2 = split_into_blocks(os.path.join(folder_path_player2, f2))[game_num][3:]
            
            for r1,r2 in zip(blocks1, blocks2):
                r1_v = r1.split(',')
                r2_v = r2.split(',')
                r1_t = r1_v[-1].strip()
                r2_t = r2_v[-1].strip()
                if r1_t == r2_t and r1_t == "r1":
                    player1_payoff += int(r1_v[-2].strip())
                elif r1_t == r2_t and r1_t == "r2":
                    player2_payoff += int(r2_v[-2].strip())
                else:
                    player1_payoff -= 0.2 * int(r1_v[-2].strip())
                    player2_payoff -= 0.2 * int(r2_v[-2].strip())
                    
            count += 1

        print(f"Analyzed {count} pairs of games of type {game_num}.")
        print(f"\tAverage payoff for Player 1: {player1_payoff/count if count>0 else 0.0}")
        print(f"\tAverage payoff for Player 2: {player2_payoff/count if count>0 else 0.0}")
        print("--------------------------------------------------")
