import itertools
from collections import Counter

def iterate_schelling_data(data):
    problems = {}
    for d in data:
        idx = d["id"]
        problems[idx] = []
        text = d["problem"]
        placeholders = [d[p] for p in d["placeholders"]]
        if not placeholders:
            problems[idx].append(text)
            continue
        
        for elements in itertools.product(*placeholders):
            new_text = text
            assert len(elements) == len(d["placeholders"]), "Mismatch in number of placeholders and elements"
            for el,p in zip(elements,d["placeholders"]):
                new_text = new_text.replace(f"@{p}@", el)
            problems[idx].append(new_text)
            
    return problems

def coordination_index(items:dict):
    freq = freq_counter = Counter(items)
    total = sum(freq_counter.values())
    coordination_index = sum([v*(v-1) for v in dict(freq).values()])/(total*(total-1))
    
    return coordination_index, coordination_index * len(freq_counter)
    
    
    
                