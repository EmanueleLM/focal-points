import itertools

def iterate_schelling_data(data):
    problems = {}
    for d in data:
        id = d["id"]
        problems[id] = []
        text = d["problem"]
        
        placeholders = [p for p in d["placeholders"]]
        if not placeholders:
            problems[id].append(text)
            continue 
        
        for elements in itertools.product(*placeholders):
            new_text = text
            assert len(elements) == len(placeholders), "Mismatch in number of placeholders and elements"
            for el,p in zip(elements,placeholders):
                new_text = new_text.replace(f"@{p}@", el)
            problems[id].append(new_text)
            
    return problems
                