import os, re, sys
site = sys.argv[1]; bad = 0; n = 0
for d, _, fs in os.walk(site):
    for f in fs:
        if not f.endswith(".html"): continue
        p = os.path.join(d, f); html = open(p, encoding="utf-8").read()
        for tag in ("<title>", 'name="viewport"', 'charset="utf-8"'):
            if tag not in html: print("MISSING", tag, p); bad += 1
        for m in re.finditer(r'(?:href|src)="([^"]+)"', html):
            h = m.group(1)
            if re.match(r"^(https?:|mailto:|#)", h): continue
            if h.startswith("/"): print("ABSOLUTE", h, p); bad += 1; continue
            tgt = h.split("#")[0]
            n += 1
            if tgt and not os.path.exists(os.path.normpath(os.path.join(d, tgt))):
                print("BROKEN", h, "in", p); bad += 1
print(f"checked {n} internal links, {bad} problems")
sys.exit(1 if bad else 0)
