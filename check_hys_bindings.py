import sys
sys.path.insert(0, r'c:\Users\jianxinlu\Repository\hys2\open-darts')

import darts.engines as eng

# List engine types
names = [n for n in dir(eng) if not n.startswith('_')]
print(f"Total exported names: {len(names)}")

# Find 2-phase compositional engine
target = None
for n in names:
    if 'cpu_fl2' in n and 'op6' in n:
        target = n
        break

if not target:
    # Try broader search
    for n in names:
        if 'fl2' in n:
            print("  fl2 engine:", n)
            target = n
            break

if target:
    print(f"\nTesting class: {target}")
    cls = getattr(eng, target)
    obj = cls()
    for attr in ['hysteresis_enabled', 'sg_max', 'Xop', 'xop_ders_arr']:
        has = hasattr(obj, attr)
        print(f"  {attr}: {'FOUND' if has else 'MISSING'}")
else:
    print("No fl2 engine found; listing all:")
    for n in names:
        print(" ", n)

# Check ms_well bindings
ms_well_cls = getattr(eng, 'ms_well', None)
if ms_well_cls:
    w = ms_well_cls()
    print("\nms_well.hysteresis_enabled:", hasattr(w, 'hysteresis_enabled'))
else:
    print("ms_well class not found in darts.engines")
