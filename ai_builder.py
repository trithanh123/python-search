import re

def get_psu_watt(psu):
    m = re.search(r'(\d+)W', psu.get('tensp', ''), re.I)
    return int(m.group(1)) if m else 0

def find_best_pc_build(budget: int, components_by_type: dict, pinned: dict = None):
    """
    Find the best PC build within budget.
    
    Args:
        budget: Maximum budget in VND.
        components_by_type: Dict of component type -> list of component dicts (sorted by semantic relevance).
        pinned: Dict of component type -> count of pinned components at top of list.
                e.g. {'CPU': 1, 'VGA': 1} means first CPU and first VGA are explicitly requested by user.
    """
    if pinned is None:
        pinned = {}

    cpus  = components_by_type.get('CPU',      [])[:5]
    mains = components_by_type.get('Mainboard',[])[:15]
    vgas  = components_by_type.get('VGA',      [])[:5]
    rams  = components_by_type.get('RAM',      [])[:3]
    ssds  = components_by_type.get('SSD',      [])[:3]
    cases = components_by_type.get('Case',     [])[:3]
    psus  = components_by_type.get('PSU',      [])[:15]

    pinned_cpu = pinned.get('CPU', 0)
    pinned_vga = pinned.get('VGA', 0)

    def _gather_builds(cpu_candidates, vga_candidates):
        """Collect all valid builds and return sorted by budget utilization (closest first)."""
        results = []
        for vga in vga_candidates:
            vga_w_str = vga.get('specifications', {}).get('power', '0W')
            try:
                vga_w = int(str(vga_w_str).replace('W', '').strip())
            except Exception:
                vga_w = 0

            psu = next((p for p in psus if get_psu_watt(p) >= vga_w), None)
            if not psu:
                continue

            for cpu in cpu_candidates:
                cpu_socket = cpu.get('specifications', {}).get('socket')
                main = next(
                    (m for m in mains if m.get('specifications', {}).get('socket') == cpu_socket),
                    None
                )
                if not main:
                    continue

                for ram in rams:
                    for ssd in ssds:
                        for case in cases:
                            build = [cpu, main, vga, ram, ssd, psu, case]
                            total_price = sum(c.get('gia', 0) for c in build)
                            if total_price <= budget:
                                results.append({
                                    'build':       build,
                                    'total_price': total_price,
                                    'diff':        budget - total_price
                                })

        results.sort(key=lambda x: x['diff'])
        return results

    # Build search phases based on which components are pinned.
    # Priority: honour user-specified components first, then loosen constraints if no match found.
    if pinned_cpu and pinned_vga:
        phases = [
            (cpus[:pinned_cpu], vgas[:pinned_vga]),  # both pinned
            (cpus[:pinned_cpu], vgas),               # pinned CPU, any VGA
            (cpus,             vgas[:pinned_vga]),   # any CPU, pinned VGA
            (cpus,             vgas),                # fallback: any
        ]
    elif pinned_cpu:
        phases = [
            (cpus[:pinned_cpu], vgas),
            (cpus,              vgas),
        ]
    elif pinned_vga:
        phases = [
            (cpus, vgas[:pinned_vga]),
            (cpus, vgas),
        ]
    else:
        phases = [(cpus, vgas)]

    for cpu_cands, vga_cands in phases:
        builds = _gather_builds(cpu_cands, vga_cands)
        if builds:
            return builds[0]['build']

    return None
