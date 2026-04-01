import subprocess
import concurrent.futures
import re
import pandas as pd

"""
JSDoc: Concurrent Evaluation Runner
@description Executes multiple RL evaluation scripts in parallel and aggregates the terminal output into a structured Pandas DataFrame.
@pattern ThreadPoolExecutor
"""


def run_single_eval(mode: str, seed: int) -> dict:
    """
    Executes eval.py for a specific mode and seed, parsing the stdout for mean and variance metrics.
    """
    command = [
        "python",
        "/content/rl_robotics/scripts/eval.py",
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]

    try:
        # We capture stdout to prevent concurrent print statements from garbling the terminal.
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        output_log = process.stdout

        # --- Early Return: Check for Empty Output ---
        if not output_log:
            return {
                "Mode": mode,
                "Seed": seed,
                "Status": "Failed: No terminal output detected.",
            }

        # --- Logic: Data Extraction via Regex ---
        # We specifically extract both the mean AND the standard deviation (+/-) to match your logs exactly.
        success_match = re.search(r"Success rate:\s+([\d.]+)%", output_log)
        reward_match = re.search(
            r"Mean reward:\s+([-]?[\d.]+)\s+\+/-\s+([\d.]+)", output_log
        )
        dist_match = re.search(r"Mean dist:\s+([\d.]+)m\s+\+/-\s+([\d.]+)m", output_log)

        # Format the extracted groups into clean strings for the table
        reward_str = (
            f"{reward_match.group(1)} ± {reward_match.group(2)}"
            if reward_match
            else "N/A"
        )
        dist_str = (
            f"{dist_match.group(1)}m ± {dist_match.group(2)}m" if dist_match else "N/A"
        )

        return {
            "Mode": mode,
            "Seed": seed,
            "Success Rate (%)": float(success_match.group(1))
            if success_match
            else None,
            "Mean Reward": reward_str,
            "Mean Dist": dist_str,
            "Status": "Success",
        }

    except subprocess.CalledProcessError as error:
        # --- Early Return: Process Failure ---
        # If MuJoCo crashes or out-of-memory occurs, we catch it here so the rest of the threads continue.
        return {
            "Mode": mode,
            "Seed": seed,
            "Success Rate (%)": None,
            "Mean Reward": None,
            "Mean Dist": None,
            "Status": f"Failed: Exit code {error.returncode}",
        }


def main():
    modes = ["curriculum", "uniform", "none"]
    seeds = [0, 1, 2]

    # Create a list of all parameter combinations
    evaluation_tasks = [(mode, seed) for mode in modes for seed in seeds]
    results_data = []

    print(f"Starting {len(evaluation_tasks)} evaluations concurrently...")

    # Set max_workers=3. Running too many MuJoCo instances simultaneously can cause RAM bottlenecks.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Map futures to their specific mode/seed tasks
        future_to_task = {
            executor.submit(run_single_eval, mode, seed): (mode, seed)
            for mode, seed in evaluation_tasks
        }

        # As each evaluation finishes, process its result
        for future in concurrent.futures.as_completed(future_to_task):
            mode, seed = future_to_task[future]

            try:
                result = future.result()
                results_data.append(result)
                print(
                    f"[Done] Mode: {mode:10} | Seed: {seed} | Status: {result['Status']}"
                )
            except Exception as e:
                print(f"[Error] Mode: {mode:10} | Seed: {seed} | Exception: {e}")

    # --- Logic: Data Formatting ---
    # Sort the results so the final table is grouped logically by Mode, then by Seed.
    sorted_results = sorted(results_data, key=lambda x: (x["Mode"], x["Seed"]))
    results_df = pd.DataFrame(sorted_results)

    # Print the aggregated markdown table
    print("\n" + "=" * 70)
    print(" TRANSFER EVALUATION SUMMARY ".center(70))
    print("=" * 70)

    # We drop the 'Status' column from the final printout if everything succeeded to keep the table clean
    if all(status == "Success" for status in results_df["Status"]):
        print(results_df.drop(columns=["Status"]).to_markdown(index=False))
    else:
        print(results_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
