import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Read the new log file
file_path = "terminal_log_for_10_minutes.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract time and NR iterations
times = []
nr_iterations = []

for line in lines:
    match = re.match(r"#\s*(\d+)\s*T\s*=\s*([\d.eE+-]+)\s*DT\s*=\s*([\d.eE+-]+)\s*NI\s*=\s*(\d+)", line)
    if match:
        time_value = float(match.group(2))
        ni_value = int(match.group(4))

        times.append(time_value * 24 * 60 * 60)   # convert day to second
        nr_iterations.append(ni_value)

# Create a dataframe
df = pd.DataFrame({"Time": times, "NR iterations": nr_iterations})

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(df["Time"], df["NR iterations"], marker=".", linestyle="--")
plt.xlabel("Simulation time [second]")
plt.ylabel("Number of NR iterations")
plt.title("NR iterations vs. Time")
plt.grid(True)
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()
