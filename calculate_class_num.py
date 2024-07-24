key_items = [0] + list(range(5, 100))

def f1(x):
    return x + 5

def f2(x):
    return x + 1

def f3(x):
    return x - 2

def f4(x):
    return x - 5

combinations = [
    lambda x: f1(f1(x)), lambda x: f1(f2(x)), lambda x: f1(f3(x)), lambda x: f1(f4(x)),
    lambda x: f2(f1(x)), lambda x: f2(f2(x)), lambda x: f2(f3(x)), lambda x: f2(f4(x)),
    lambda x: f3(f1(x)), lambda x: f3(f2(x)), lambda x: f3(f3(x)), lambda x: f3(f4(x)),
    lambda x: f4(f1(x)), lambda x: f4(f2(x)), lambda x: f4(f3(x)), lambda x: f4(f4(x))
]

unique_results = set()

for key_item in key_items:
    for combo in combinations:
        unique_results.add(combo(key_item))

print("Number of unique results:", len(unique_results))
print("Unique results:", unique_results)
