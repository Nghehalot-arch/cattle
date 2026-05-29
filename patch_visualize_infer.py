from pathlib import Path

p = Path("visualize_json_results_infer.py")
s = p.read_text()

old = '        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])'

new = '''        if len(pred_by_image[dic["image_id"]]) == 0:
            continue
        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])'''

if old not in s:
    print("Could not find the exact line. Printing nearby lines:")
    lines = s.splitlines()
    for i, line in enumerate(lines, start=1):
        if "create_instances" in line:
            print(i, repr(line))
else:
    s = s.replace(old, new)
    p.write_text(s)
    print("Patched visualize_json_results_infer.py")
