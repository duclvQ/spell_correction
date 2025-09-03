
import re 

def intersect(pos1, pos2):
    """Check if two position ranges intersect"""
    if pos1[0] < pos2[1] and pos2[0] < pos1[1]:
        return True
    if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
        return True
    # inside
    if pos1[0] < pos2[0] and pos1[1] > pos2[1]:
        return True
    if pos1[0] > pos2[0] and pos1[1] < pos2[1]:
        return True
    return False

def apply_ops_with_offset(text, merged_replacements):
    chars = list(text)
    # sort by start ascending
    merged_replacements = sorted(merged_replacements, key=lambda r: r['pos'][0])

    cleaned_replacements = []
    # remove redundant replacements
    for i in range(len(merged_replacements)):
        if len(cleaned_replacements)<1:
            cleaned_replacements.append(merged_replacements[i])
        else:
            for c in cleaned_replacements:
                if intersect(c['pos'], merged_replacements[i]['pos']):
                    # keep the one with the longer length
                    if c['pos'][1] - c['pos'][0] > merged_replacements[i]['pos'][1] - merged_replacements[i]['pos'][0]:
                        merged_replacements[i]['op'] = "noop"
                        pass
                    else:
                        # c['op'] = "noop"
                        pass 
                else:
                    cleaned_replacements.append(merged_replacements[i])
    merged_replacements = cleaned_replacements

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
