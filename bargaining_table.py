
# List all the files in a folder
import math
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
                # Start a new block
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

def extract_svo_angle(csv_path: str) -> float:
    """
    Extracts the Player SVO angle (in degrees) from a CSV file.

    The function assumes the SVO angle is stored on a separate line
    as the first value, followed by descriptive text.
    """
    with open(csv_path, "r") as f:
        for line in f:
            if "SVO angle" in line:
                return float(line.split(",")[0])

    raise ValueError("SVO angle not found in file.")

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

def compute_payoff(
    sampled_pairs: List[Tuple[str, str]],
    folder_path_player1: str,
    folder_path_player2: str,
    strategy: str = "humans"
) -> dict:
    """
    Computes the payoff for both players based on the sampled pairs of games and the chosen strategy.

    Args:
        sampled_pairs (List[Tuple[str, str]]): The sampled pairs of game files.
        folder_path_player1 (str): The folder path for player 1's game files.
        folder_path_player2 (str): The folder path for player 2's game files.
        strategy (str, optional): The strategy to use for computing payoffs (defaults to "humans"). 
            The strategy can be:
            - "humans" (based on data for both players), 
            - "p1_greedy" (player 1 is greedy, player 2 is based on his data), 
            - "p2_greedy" (player 2 is greedy, player 1 is based on his data), 
            - "both_greedy" (not very interesting, both players are greedy),
            - "p1_cooperative" (player 1 chooses the assignments based on the Euclidean distance from the disk).
            - "p2_cooperative" (player 2 chooses the assignments based on the Euclidean distance from the disk).
            - "p1_SVO" (player 1 chooses the assignments based on Social Value Orientation). Given the angle a, computes 
            t1 = cos(a)*v1 + sin(a)*v2 and t2 = sin(a)*v1 + cos(a)*v2, where vi is the utility if they keep the disk. 
            Then assigns to himself if t1 > t2, otherwise to the other.
            - "p2_SVO" (player 2 chooses the assignments based on Social Value Orientation). Same but for player 2.

    Raises:
        ValueError: If the game signatures do not match.
        ValueError: If the strategy is unknown.

    Returns:
        dict: A dictionary containing the total payoffs for both players.
    """
    player1_total_payoff, player2_total_payoff = 0.0, 0.0
    for game_num in range(1,11):
        count = 0
        player1_payoff, player2_payoff = 0.0, 0.0
        total_choices, suboptimal_choices = 0, 0
        pay_off_left = 0.0
        for f1, f2 in sampled_pairs:
            sign_f1 = game_signature(os.path.join(folder_path_player1, f1))
            sign_f2 = game_signature(os.path.join(folder_path_player2, f2))
            if sign_f1 != sign_f2:
                raise ValueError(f"Game signatures do not match for files {f1} and {f2}!")

            whole_block1 = split_into_blocks(os.path.join(folder_path_player1, f1))[game_num]
            whole_block2 = split_into_blocks(os.path.join(folder_path_player2, f2))[game_num]
            
            # Positions of the players in "cooperative" strategy
            if strategy in ["p1_cooperative", "p2_cooperative"]:
                p1x = int(whole_block1[1].split(',')[-2].strip())
                p1y = int(whole_block1[1].split(',')[-1].strip())
                p2x = int(whole_block2[2].split(',')[-2].strip())
                p2y = int(whole_block2[2].split(',')[-1].strip())
                # print(os.path.join(folder_path_player1, f1))
                # print(os.path.join(folder_path_player2, f2))
                # print(f"Player 1 position: ({p1x},{p1y}), Player 2 position: ({p2x},{p2y})")
            
            blocks1 = whole_block1[3:]  # Skip first 3 rows (header)
            blocks2 = whole_block2[3:]  # Skip first 3 rows (header)
            
            # Iterate through corresponding blocks of both players, according to the strategy
            for r1,r2 in zip(blocks1, blocks2):
                if strategy == "humans":
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    r1_t = r1_v[-1].strip()
                    r2_t = r2_v[-1].strip()
                    
                elif strategy in ["p1_greedy", "p2_greedy", "both_greedy"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    if strategy == "p1_greedy":
                        r1_t = "r1"  # Player 1 always chooses r1
                        r2_t = r2_v[-1].strip()
                    elif strategy == "p2_greedy":
                        r1_t = r1_v[-1].strip()
                        r2_t = "r2"  # Player 2 always chooses r2
                    elif strategy == "both_greedy":
                        r1_t = "r1"  # Player 1 always chooses r1
                        r2_t = "r2"  # Player 2 always chooses r2
                        
                elif strategy in ["p1_cooperative", "p2_cooperative"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    # Disk position
                    dx = int(r1_v[-4].strip())
                    dy = int(r1_v[-3].strip())
                    dist_p1_d = ((p1x - dx)**2 + (p1y - dy)**2)**0.5
                    dist_p2_d = ((p2x - dx)**2 + (p2y - dy)**2)**0.5
                    if strategy == "p1_cooperative":
                        if dist_p1_d >= dist_p2_d:
                            r1_t = "r2"
                        else:
                            r1_t = "r1"
                        r2_t = r2_v[-1].strip()
                    else:  # p2_cooperative
                        r1_t = r1_v[-1].strip()
                        if dist_p2_d >= dist_p1_d:
                            r2_t = "r1"
                        else:
                            r2_t = "r2"
                        r1_t = r1_v[-1].strip()
                    
                elif strategy in ["p1_SVO", "p2_SVO"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    v1_r1, v2_r1 = int(r1_v[-2].strip()), int(r2_v[-2].strip())
                    # SVO angle fixed to 22.5 degrees
                    angle_rad = 22.5 * (math.pi / 180)
                    threshold_p1 = math.cos(angle_rad) * v1_r1 # + math.sin(angle_rad) * 0.
                    threshold_p2 = math.sin(angle_rad) * v2_r1  # + math.cos(angle_rad) * 0.
                    if strategy == "p1_SVO":
                        if threshold_p1 > threshold_p2:
                            r1_t = "r1"
                        else:
                            r1_t = "r2"
                        r2_t = r2_v[-1].strip()
                    else:
                        if threshold_p1 > threshold_p2:
                            r2_t = "r2"
                        else:
                            r2_t = "r1"
                        r1_t = r1_v[-1].strip()
                        
                else:
                    raise ValueError(f"Unknown mode: {strategy}")

                # Assign payoffs based on each player choices
                if r1_t == r2_t == "r1":
                    player1_payoff += int(r1_v[-2].strip())
                elif r1_t == r2_t == "r2":
                    player2_payoff += int(r2_v[-2].strip())
                else:
                    player1_payoff -= 0.2 * int(r1_v[-2].strip())
                    player2_payoff -= 0.2 * int(r2_v[-2].strip())
                    pay_off_left += int(r1_v[-2].strip())
                    suboptimal_choices += 1
                total_choices += 1
                
            count += 1
            
        player1_total_payoff += player1_payoff
        player2_total_payoff += player2_payoff
        
        print("--------------------------------------------------")
        print(f"Strategy: {strategy}")
        print(f"Analyzed {count} pairs of games of type {game_num}.")
        print(f"\tAverage payoff for Player 1: {player1_payoff/count if count>0 else 0.0}")
        print(f"\tAverage payoff for Player 2: {player2_payoff/count if count>0 else 0.0}")
        print(f"\tNumber of suboptimal choices: {suboptimal_choices}/{total_choices} ({(suboptimal_choices/total_choices*100) if total_choices > 0 else 0} %)")
        print(f"\tAverage payoff left: {pay_off_left/count if count>0 else 0.0}")
        print(f"\tPayoff ratio (P1/P2): {(player1_payoff/count)/(player2_payoff/count) if player2_payoff/count != 0 else 'inf'}")
        print(f"\tPayoff ratio (P2/P1): {(player2_payoff/count)/(player1_payoff/count) if player1_payoff/count != 0 else 'inf'}")
    print(f"Total payoff - across games - Player 1: {player1_total_payoff/(count)}, Player 2: {player2_total_payoff/(count)}")
    print("--------------------------------------------------")

    return {
        "mode": strategy,
        "count": count,
        "player1_payoff": player1_payoff,
        "player2_payoff": player2_payoff,
        "suboptimal_choices": suboptimal_choices,
        "total_choices": total_choices,
        "pay_off_left": pay_off_left
    }

if __name__ == "__main__":
    """
    Strategy can be:
        - "humans" (based on data for both players), 
        - "p1_greedy" (player 1 is greedy, player 2 is based on his data), 
        - "p2_greedy" (player 2 is greedy, player 1 is based on his data), 
        - "both_greedy" (not very interesting, both players are greedy),
        - "p1_cooperative" (player 1 chooses the assignments based on the Euclidean distance from the disk).
        - "p2_cooperative" (player 2 chooses the assignments based on the Euclidean distance from the disk).
        - "p1_SVO" (player 1 chooses the assignments based on Social Value Orientation).
        - "p2_SVO" (player 2 chooses the assignments based on Social Value Orientation).
    """
    # Global parameters
    strategy = "p1_SVO"  # Strategy to use
    num_samples = 1000  # Number of random pairs to sample
    sample_with_replacement = False  # Whether to sample with replacement
    folder_path_player1 = "./data/Dor-humans/stage-2-analysis/Number1players/"
    folder_path_player2 = "./data/Dor-humans/stage-2-analysis/Number2players/"

    # Retrieve all the games played by both players
    player1_files = player_files(folder_path_player1, "BT")
    player2_files = player_files(folder_path_player2, "BT")
    
    sampled_pairs = sample_pairs(
        player1_files,
        player2_files,
        num_samples,
        with_replacement=sample_with_replacement,
    )

    results = compute_payoff(
        sampled_pairs,
        folder_path_player1,
        folder_path_player2,
        strategy=strategy
    )
