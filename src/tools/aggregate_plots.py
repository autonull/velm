import os
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
outdirs = sorted([d for d in glob.glob(os.path.join(ROOT, 'outputs*')) if os.path.isdir(d)])
if not outdirs:
    print('No outputs directories found')
    raise SystemExit(0)

summary = []
for d in outdirs:
    name = os.path.basename(d)
    pre = None
    post = None
    prefile = os.path.join(d, 'prepost.txt')
    postfile = os.path.join(d, 'eggroll_result.txt')
    if os.path.exists(prefile):
        with open(prefile) as f:
            for l in f:
                m = re.search(r'pre_acc=([0-9.eE+-]+)', l)
                if m:
                    pre = float(m.group(1))
    if os.path.exists(postfile):
        with open(postfile) as f:
            for l in f:
                m = re.search(r'eggroll_post_acc=([0-9.eE+-]+)', l)
                if m:
                    post = float(m.group(1))
    summary.append((name, pre, post, d))

# Save CSV-like summary
csv_path = os.path.join(ROOT, 'outputs_summary.csv')
with open(csv_path, 'w') as f:
    f.write('run,pre_acc,post_acc,dir\n')
    for name, pre, post, d in summary:
        f.write(f'{name},{pre if pre is not None else ""},{post if post is not None else ""},{d}\n')
print('Wrote', csv_path)

# Bar plot of pre/post for runs that have values
names = []
pre_vals = []
post_vals = []
for name, pre, post, d in summary:
    if pre is None and post is None:
        continue
    names.append(name)
    pre_vals.append(pre if pre is not None else 0.0)
    post_vals.append(post if post is not None else 0.0)

if names:
    x = range(len(names))
    plt.figure(figsize=(max(6, len(names)*1.2),4))
    width = 0.35
    plt.bar([i-width/2 for i in x], pre_vals, width=width, label='pre')
    plt.bar([i+width/2 for i in x], post_vals, width=width, label='post')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('accuracy')
    plt.title('Pre vs Post EGGROLL accuracy')
    plt.legend()
    outp = os.path.join(ROOT, 'outputs', 'summary_prepost.png')
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp)
    print('Saved', outp)
else:
    print('No pre/post data to plot')

# Also produce per-run simple report and save any loss.png or prepost_acc.png into a collated folder
collate = os.path.join(ROOT, 'outputs', 'collated')
os.makedirs(collate, exist_ok=True)
for name, pre, post, d in summary:
    for fn in ['loss.png', 'prepost_acc.png']:
        src = os.path.join(d, fn)
        if os.path.exists(src):
            dst = os.path.join(collate, f'{name}_{fn}')
            try:
                from shutil import copyfile
                copyfile(src, dst)
            except Exception:
                pass
print('Collated available images to', collate)
