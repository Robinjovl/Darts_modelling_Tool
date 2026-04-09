import darts.engines as eng

# Check engine_base (abstract - use a concrete subclass)
# Try 2-phase super engines
for cls_name in ['engine_super_cpu2_1', 'engine_super_cpu2_2', 'engine_nc_nl_cpu2', 'engine_super_cpu2_3']:
    cls = getattr(eng, cls_name, None)
    if cls:
        obj = cls()
        print(f"=== {cls_name} ===")
        for attr in ['hysteresis_enabled', 'sg_max', 'Xop', 'xop_ders_arr']:
            has = hasattr(obj, attr)
            print(f"  {attr}: {'FOUND' if has else 'MISSING'}")
        break

# Check interpolator with 6 dims, 11 ops (ABPair<6,11> from py_operator_set_interpolator_super.cpp)
print("\n=== Checking for 6_11 interpolator (ABPair<6,11>) ===")
found = False
for n in dir(eng):
    if '6_11' in n:
        print(f"  FOUND: {n}")
        found = True
if not found:
    print("  NOT FOUND - ABPair<6,11> not compiled in")

# ms_well check
w = eng.ms_well()
print(f"\n=== ms_well ===")
print(f"  hysteresis_enabled: {'FOUND' if hasattr(w, 'hysteresis_enabled') else 'MISSING'} = {getattr(w, 'hysteresis_enabled', 'N/A')}")
