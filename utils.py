
import re 


def apply_ops_with_offset(text, merged_replacements):
    chars = list(text)
    # sort by start ascending
    merged_replacements = sorted(merged_replacements, key=lambda r: r['pos'][0])

    offset = 0
    for r in merged_replacements:
        op   = r['op']
        (s, e) = r['pos']
        orig = r.get('orig', text[s:e])
        repl = r.get('repl', "")

        s_adj = s + offset
        e_adj = e + offset
        print("s_adj", s_adj)
        print("e_adj", e_adj)
        print("offset", offset)
        if op == "insert":
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
            raise ValueError(f"Unknown op: {op}")
        print("len tag", len(tag))
        print("tag", tag)
        print("orig", orig)
        print("repl", repl)
        print("delta", delta)
        print("chars", chars)
        offset += delta  # shift subsequent ops

    return "".join(chars)
