import inspect

try:
    import depth_loader
    print("=== depth_loader.py contents ===")
    print(inspect.getsource(depth_loader))
except Exception as e:
    print(f"Could not import depth_loader: {e}")