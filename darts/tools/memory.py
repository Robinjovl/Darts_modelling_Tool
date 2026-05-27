
def get_peak_memory_in_bytes():
    try:   
        import psutil
        import platform
        current_os = platform.system()
        
        if current_os == "Windows":
            # psutil returns peak_wset on Windows in Bytes
            p = psutil.Process(os.getpid())
            return p.memory_info().peak_wset
        elif current_os == "Linux":
            # resource works on Linux, but returns Kilobytes
            import resource
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak_kb * 1024  # Convert KB to Bytes
        elif current_os == "Darwin":  # macOS
            import resource
            # macOS returns ru_maxrss in Bytes
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    except ImportError:
        print("Memory info unavailable (psutil, resource and platform modules are required)")

    return None

def print_allocated_memory():
    GB2B =  1024 ** 3
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / GB2B
        peak = get_peak_memory_in_bytes() / GB2B
        vms = proc.memory_info().vms / GB2B
        print(f"Memory usage: RSS = {rss:.2f} GB, Peak RSS = {peak:.2f} GB, VMS = {vms:.2f} GB")
    except ImportError:
        print("Memory info unavailable (psutil module is required)")
