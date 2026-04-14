"""
Training log health analyzer.
Usage: python scripts/analyze_log.py /path/to/pretrain.log
"""
import re
import sys
from statistics import mean, stdev

def parse_log(path):
    steps, vals, cores = [], [], []
    with open(path) as f:
        for line in f:
            # Training step
            m = re.match(
                r'step (\d+)/(\d+) .* loss: ([\d.]+) \| lrm: ([\d.]+) \| '
                r'dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| mfu: ([\d.]+) \| '
                r'epoch: (\d+)',
                line
            )
            if m:
                steps.append(dict(
                    step=int(m[1]), total=int(m[2]),
                    loss=float(m[3]), lrm=float(m[4]),
                    dt=float(m[5]), tok_sec=int(m[6].replace(',', '')),
                    mfu=float(m[7]), epoch=int(m[8]),
                ))
                continue
            # Validation bpb
            m = re.match(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
            if m:
                vals.append(dict(step=int(m[1]), bpb=float(m[2])))
                continue
            # CORE metric
            m = re.match(r'Step (\d+) \| CORE metric: ([\d.]+)', line)
            if m:
                cores.append(dict(step=int(m[1]), core=float(m[2])))
    return steps, vals, cores

def analyze(path):
    steps, vals, cores = parse_log(path)
    if not steps:
        print("No training steps found in log.")
        return

    last = steps[-1]
    total_steps = last['total']
    issues = []

    # --- Progress ---
    print(f"{'='*60}")
    print(f"  TRAINING HEALTH REPORT")
    print(f"{'='*60}")
    print(f"  Steps: {last['step']:,} / {total_steps:,} ({100*last['step']/total_steps:.2f}%)")
    print(f"  Epoch: {last['epoch']}")
    # find total time and eta from the raw line
    losses = [s['loss'] for s in steps]
    mfus = [s['mfu'] for s in steps]
    dts = [s['dt'] for s in steps]
    tok_secs = [s['tok_sec'] for s in steps]

    # --- Loss ---
    print(f"\n--- Loss ---")
    print(f"  Current (EMA): {losses[-1]:.6f}")
    print(f"  Min:           {min(losses):.6f} (step {steps[losses.index(min(losses))]['step']})")
    print(f"  Max:           {max(losses):.6f} (step {steps[losses.index(max(losses))]['step']})")

    # Trend: compare last 200 steps vs previous 200
    if len(losses) > 400:
        recent = mean(losses[-200:])
        previous = mean(losses[-400:-200])
        pct = 100 * (recent - previous) / previous
        direction = "decreasing" if pct < 0 else "INCREASING"
        print(f"  Trend:         {direction} ({pct:+.2f}% last 200 vs prior 200)")
        if pct > 5:
            issues.append(f"Loss INCREASING by {pct:.1f}%")
    elif len(losses) > 100:
        half = len(losses) // 2
        recent = mean(losses[half:])
        previous = mean(losses[:half])
        pct = 100 * (recent - previous) / previous
        direction = "decreasing" if pct < 0 else "INCREASING"
        print(f"  Trend:         {direction} ({pct:+.2f}%)")

    # Spike detection: any loss > 3x the local rolling average
    window = 50
    spikes = []
    for i in range(window, len(losses)):
        local_avg = mean(losses[i-window:i])
        if losses[i] > 3 * local_avg:
            spikes.append((steps[i]['step'], losses[i], local_avg))
    if spikes:
        print(f"  SPIKES:        {len(spikes)} detected (loss > 3x local avg)")
        for s, l, a in spikes[-5:]:  # show last 5
            print(f"    step {s}: loss={l:.4f} (avg={a:.4f}, {l/a:.1f}x)")
        issues.append(f"{len(spikes)} loss spike(s)")
    else:
        print(f"  Spikes:        none")

    # --- Throughput ---
    print(f"\n--- Throughput ---")
    # Skip first 10 steps (warmup/compilation)
    warm = steps[min(10, len(steps)):]
    if warm:
        w_mfus = [s['mfu'] for s in warm]
        w_dts = [s['dt'] for s in warm]
        w_tok = [s['tok_sec'] for s in warm]
        print(f"  MFU:           {mean(w_mfus):.2f}% (std={stdev(w_mfus):.2f}%)" if len(w_mfus) > 1 else f"  MFU:           {w_mfus[0]:.2f}%")
        print(f"  tok/sec:       {mean(w_tok):,.0f} (std={stdev(w_tok):,.0f})" if len(w_tok) > 1 else f"  tok/sec:       {w_tok[0]:,}")
        print(f"  dt:            {mean(w_dts):.1f}ms (std={stdev(w_dts):.1f}ms)" if len(w_dts) > 1 else f"  dt:            {w_dts[0]:.1f}ms")
        # Check for throughput degradation
        if len(w_mfus) > 200:
            early_mfu = mean(w_mfus[:100])
            late_mfu = mean(w_mfus[-100:])
            if late_mfu < early_mfu * 0.9:
                issues.append(f"MFU degraded: {early_mfu:.1f}% -> {late_mfu:.1f}%")
                print(f"  WARNING:       MFU degraded from {early_mfu:.1f}% to {late_mfu:.1f}%")

    # --- Validation BPB ---
    if vals:
        print(f"\n--- Validation BPB ---")
        for v in vals:
            print(f"  Step {v['step']:>6d}: {v['bpb']:.6f}")
        if len(vals) >= 2:
            direction = "improving" if vals[-1]['bpb'] < vals[-2]['bpb'] else "WORSENING"
            print(f"  Trend:         {direction}")
            if vals[-1]['bpb'] > vals[-2]['bpb']:
                issues.append(f"Val BPB worsening: {vals[-2]['bpb']:.4f} -> {vals[-1]['bpb']:.4f}")

    # --- CORE Metric ---
    if cores:
        print(f"\n--- CORE Metric ---")
        for c in cores:
            print(f"  Step {c['step']:>6d}: {c['core']:.4f}")
        if len(cores) >= 2:
            direction = "improving" if cores[-1]['core'] > cores[-2]['core'] else "DECLINING"
            print(f"  Trend:         {direction}")

    # --- LR Schedule ---
    print(f"\n--- LR Schedule ---")
    print(f"  Current lrm:   {last['lrm']:.4f}")
    if last['lrm'] < 1.0:
        warmdown_start = total_steps * 0.6  # default warmdown_ratio=0.4
        if last['step'] >= warmdown_start:
            print(f"  Status:        in warmdown phase (expected)")
        else:
            print(f"  WARNING:       lrm < 1.0 before warmdown")
            issues.append("LR dropped before warmdown phase")

    # --- Verdict ---
    print(f"\n{'='*60}")
    if not issues:
        print(f"  VERDICT: GREEN - Training looks healthy")
    elif any("INCREASING" in i or "spike" in i for i in issues):
        print(f"  VERDICT: RED - Issues detected:")
    else:
        print(f"  VERDICT: YELLOW - Minor concerns:")
    for i in issues:
        print(f"    - {i}")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)
    analyze(sys.argv[1])
