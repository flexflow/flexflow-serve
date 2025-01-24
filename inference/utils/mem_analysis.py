import pandas as pd
import re, os, math, argparse

# Usage:
# Run FlexFlow code with --log-instance-creation flag and redirect the output to a file
# python mem_analysis.py --file_path /path/to/log_file.txt

def extract_data(file_path):
    # Define regex patterns
    memory_allocator_pattern = re.compile(r'MemoryAllocator.*memory_kind: (\w+).*memory_id: (\w+).*size: (\d+).*capacity (\d+).*task_name: (.+)')
    mapper_pattern = re.compile(r'Mapper.*memory_kind: (\w+).*memory_id: (\w+).*size: (\d+).*capacity (\d+).*task: (.+)')
    parallel_tensor_pattern = re.compile(r'ParallelTensor.*memory_kind: (\w+).*memory_id: (\w+).*size: (\d+).*capacity (\d+).*task_name: (.+)')

    # Initialize lists to store extracted data
    memory_kinds = []
    memory_ids = []
    sizes = []
    capacities = []
    tasks = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            if 'MemoryAllocator' in line:
                match = memory_allocator_pattern.search(line)
                if match:
                    memory_kinds.append(match.group(1))
                    memory_ids.append(match.group(2))
                    sizes.append(int(match.group(3)))
                    capacities.append(int(match.group(4)))
                    tasks.append(match.group(5))
            elif 'Mapper' in line:
                match = mapper_pattern.search(line)
                if match:
                    memory_kinds.append(match.group(1))
                    memory_ids.append(match.group(2))
                    sizes.append(int(match.group(3)))
                    capacities.append(int(match.group(4)))
                    tasks.append(match.group(5))
            elif 'ParallelTensor' in line:
                match = parallel_tensor_pattern.search(line)
                if match:
                    memory_kinds.append(match.group(1))
                    memory_ids.append(match.group(2))
                    sizes.append(int(match.group(3)))
                    capacities.append(int(match.group(4)))
                    tasks.append(match.group(5))

    # Create a DataFrame
    df = pd.DataFrame({
        'Memory Kind': memory_kinds,
        'Device ID': memory_ids,
        'Size': sizes,
        'Capacity': capacities,
        'Task': tasks
    })

    return df

def human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1000)))
    p = math.pow(1000, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def print_grouped_by_device(df):
    grouped_df = df.groupby(['Memory Kind', 'Device ID']).agg({'Size': 'sum', 'Capacity': 'first'})
    # Check that all entries that share the same memory id have the same capacity
    for (memory_kind, memory_id), group in df.groupby(['Memory Kind', 'Device ID']):
        capacities = group['Capacity'].unique()
        if len(capacities) > 1:
            print(f"Warning: Device ID {memory_id} in Memory Kind {memory_kind} has multiple capacities: {capacities}")
    # Convert sizes to human-readable format
    grouped_df['Size'] = grouped_df['Size'].apply(human_readable_size)
    grouped_df['Capacity'] = grouped_df['Capacity'].apply(human_readable_size)
    print("############## Memory usage (by device) ##############")
    print(grouped_df)

def print_grouped_by_task(df):
    # Group by 'Memory Kind', 'Device ID', and 'Task', and sum the 'Size' column
    task_grouped_df = df.groupby(['Memory Kind', 'Device ID', 'Task']).agg({'Size': 'sum'}).reset_index()
    # Sort the DataFrame by 'Memory Kind', 'Device ID', and 'Size' in descending order
    task_grouped_df = task_grouped_df.sort_values(by=['Memory Kind', 'Device ID', 'Size'], ascending=[True, True, False])
    print("\n\n############## Memory usage (by task) ##############")
    for (memory_kind, memory_id), group in task_grouped_df.groupby(['Memory Kind', 'Device ID']):
        print("\n-------------------------------------------------------------")
        print(f"Memory Kind: {memory_kind}, Device ID: {memory_id}")
        group['Size'] = group['Size'].apply(human_readable_size)
        print(group[['Task', 'Size']].to_string(index=False))
        print("-------------------------------------------------------------")

def print_notes():
    print("\n\n############## Notes ##############")
    print("* Check that each GPU retains enough capacity in GPU_FB_MEM to hold the weights from Z_COPY_MEM (total size / tp_degree)")
    print("* Check whether the memory usage is balanced across devices")
    print("* `set_tensor` generally refers to the memory used to load the model weights")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze memory usage from a FlexFlow log file.')
    parser.add_argument('--file_path', '-fp', type=str, help='Path to the input log file')
    args = parser.parse_args()

    # Change working directory to the directory holding the script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)
    
    df = extract_data(args.file_path)
    print_grouped_by_device(df)
    print_grouped_by_task(df)

    print_notes()