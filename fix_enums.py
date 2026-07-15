import os

replacements = {
    'TraceEventType.STAGE_START': 'TraceEventType.QUERY_LOAD',
    'TraceEventType.SCOPE_FILTERED': 'TraceEventType.SCOPE_ENUMERATE',
    'TraceEventType.FACTS_EXTRACTED': 'TraceEventType.FACT_EXTRACT',
    'TraceEventType.AGGREGATION_COMPLETE': 'TraceEventType.AGGREGATE',
    'TraceEventType.STAGE_END': 'TraceEventType.PERSIST',
}

d = 'packages/pipelines/faulttrace_pipelines'
for f in os.listdir(d):
    if f.startswith('p') and f.endswith('.py') and f != 'p0_baseline.py':
        p = os.path.join(d, f)
        with open(p, 'r', encoding='utf-8') as file:
            c = file.read()
            
        for k, v in replacements.items():
            c = c.replace(k, v)
            
        c = c.replace('"validate", TraceEventType.QUERY_LOAD', '"validate", TraceEventType.VALIDATE')
        
        with open(p, 'w', encoding='utf-8') as file:
            file.write(c)
