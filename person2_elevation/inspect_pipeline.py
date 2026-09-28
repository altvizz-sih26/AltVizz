import inspect
import elevation_pipeline as ep

print("=== generate_elevation signature ===")
print(inspect.signature(ep.generate_elevation))
print()
print("=== generate_elevation source ===")
print(inspect.getsource(ep.generate_elevation))