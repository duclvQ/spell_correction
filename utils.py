
import re 

def intersects(a, b):
    # Half-open intervals [start, end): overlap iff starts are before the other's end
    return a[0] < b[1] and b[0] < a[1]

def span_len(pos):
    return pos[1] - pos[0]

def filter_overlaps_keep_longest(repls):
    """
    repls: list[{'pos': (start, end), ...}]
    Returns a new list with no overlaps, keeping the longest item inside each
    overlapping cluster. If lengths tie, keeps the earliest (smallest start, then smallest end).
    """
    if not repls:
        return []

    # sort by start then end for stable grouping
    items = sorted(repls, key=lambda r: (r['pos'][0], r['pos'][1]))
    result = []

    # start first group
    group = [items[0]]
    group_max_end = items[0]['pos'][1]

    for itm in items[1:]:
        if itm['pos'][0] < group_max_end and intersects(itm['pos'], group[-1]['pos']):
            # still overlapping with current group
            group.append(itm)
            group_max_end = max(group_max_end, itm['pos'][1])
        else:
            # close previous group -> pick the best
            best = max(group, key=lambda r: (span_len(r['pos']), -r['pos'][0], -r['pos'][1]))
            result.append(best)
            # start new group
            group = [itm]
            group_max_end = itm['pos'][1]

    # close last group
    best = max(group, key=lambda r: (span_len(r['pos']), -r['pos'][0], -r['pos'][1]))
    result.append(best)

    return result
def filter_overlaps_keep_shortest(repls):
    """
    Keep the SHORTEST item in each overlapping cluster.
    Tie-break: earliest start, then earliest end.
    """
    if not repls:
        return []

    items = sorted(repls, key=lambda r: (r['pos'][0], r['pos'][1]))
    result = []

    group = [items[0]]
    group_max_end = items[0]['pos'][1]

    for itm in items[1:]:
        if itm['pos'][0] < group_max_end and intersects(itm['pos'], group[-1]['pos']):
            group.append(itm)
            group_max_end = max(group_max_end, itm['pos'][1])
        else:
            best = min(group, key=lambda r: (span_len(r['pos']), r['pos'][0], r['pos'][1]))
            result.append(best)
            group = [itm]
            group_max_end = itm['pos'][1]

    best = min(group, key=lambda r: (span_len(r['pos']), r['pos'][0], r['pos'][1]))
    result.append(best)

    return result
def mark_overlaps_noop(merged_replacements):
    keep = filter_overlaps_keep_longest(merged_replacements)
    keep_ids = {id(x) for x in keep}  # identity to avoid deep equality pitfalls
    cleaned = []
    for r in merged_replacements:
        if id(r) in keep_ids:
            cleaned.append(r)
        else:
            # copy so original input isn't mutated (optional)
            rr = dict(r)
            rr['op'] = 'noop'
            cleaned.append(rr)
    return cleaned


def apply_ops_with_offset(text, merged_replacements):
    chars = list(text)
    # sort by start ascending
    merged_replacements = sorted(merged_replacements, key=lambda r: r['pos'][0])

    clean_replacements = filter_overlaps_keep_shortest(merged_replacements)
    merged_replacements = clean_replacements
    offset = 0
    for r in merged_replacements:
        op   = r['op']
        (s, e) = r['pos']
        print("s", s)
        print("e", e)
        orig = r.get('orig', text[s:e])
        repl = r.get('repl', "")
        

        s_adj = s + offset
        e_adj = e + offset
        print("s_adj", s_adj)
        print("e_adj", e_adj)
        print("offset", offset)
        if op == "insert":
            # skip if before and after is digit
            if s_adj > 0 and e_adj < len(chars) and chars[s_adj-1].isdigit() or chars[e_adj].isdigit():
                continue
            tag = f"<insert|{repl}|>"
            chars[s_adj:s_adj] = tag
            delta = len(tag)  # inserted length
        elif op == "delete":
            tag = f"<delete|{orig}|>"
            old_len = e_adj - s_adj
            chars[s_adj:e_adj] = tag
            delta = len(tag) - old_len
        elif op == "replace":
            tag = f"<replace|{repl}|>"
            old_len = e_adj - s_adj
            chars[s_adj:e_adj] = tag
            delta = len(tag) - old_len
        else:
            # raise ValueError(f"Unknown op: {op}")
            print("Unknown op:", op)
            pass

        print("len tag", len(tag))
        print("tag", tag)
        print("orig", orig)
        print("repl", repl)
        print("delta", delta)
        print("chars", chars)
        offset += delta  # shift subsequent ops

    return "".join(chars)
