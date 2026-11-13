#!/usr/bin/env python3
# Render the bench prompts' messages through the official Qwen3.8 chat template (add_generation_prompt) into raw prompt files.
import json, sys, jinja2
T = '/path/to/project/traces/qwen38/qwen38-chat-template.jinja'
P = '/path/to/project/bench/prompts/flash/'
O = '/path/to/project/traces/qwen38/rec/'
def raise_exception(m): raise RuntimeError(m)
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
env.globals['raise_exception'] = raise_exception
import re
# {% generation %} marks assistant tokens for training (Hugging Face's tag); it does not change the text
src = re.sub(r'{%-? *(end)?generation *-?%}', '', open(T).read())
tpl = env.from_string(src)
for n in sys.argv[1:]:
    msgs = json.load(open(P + n + '.messages.json'))
    s = tpl.render(messages=msgs, add_generation_prompt=True, bos_token='', eos_token='')
    open(O + n + '.txt', 'w').write(s)
    print(n, repr(s[:60]), '...', repr(s[-40:]))
