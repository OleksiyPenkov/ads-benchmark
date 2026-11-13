import json
import random

from tool_schemas import schemas_for

SMOKE_TOOLS = ["describe_machine", "system_status", "tail_log"]
SMOKE_VOCAB_SIZE = 1024
SMOKE_MAX_POSITIONS = 4096

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]

TEMPLATE_STYLE = {"qwen3-32b": "json", "qwen3.8-moe": "xml"}

JSON_STYLE_TEMPLATE = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content or '' %}
        {%- if loop.index0 > ns.last_query_index and loop.last %}
            {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content.lstrip('\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

XML_STYLE_TEMPLATE = r"""
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = messages[0].content|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content|trim + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = (message.content or '')|trim %}
    {%- if message.role == "system" %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"""

TEMPLATES = {"json": JSON_STYLE_TEMPLATE, "xml": XML_STYLE_TEMPLATE}

SYSTEM_PROMPT = "You operate a thin-film deposition system through tools. Read state before you answer."


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant_calls(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def _tool_result(call_id, name, payload):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(payload)}


def _read_case(rng):
    count = rng.choice([3, 5, 10])
    entries = [{"t": 100 + i, "event": rng.choice(["valve_open", "pump_on", "ramp_done"])} for i in range(count)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Show the last {count} log entries."},
        _assistant_calls(_call("call_1", "tail_log", {"count": count})),
        _tool_result("call_1", "tail_log", {"entries": entries}),
        {"role": "assistant", "content": f"The last {count} events: " + ", ".join(e["event"] for e in entries) + "."},
    ]


def _parallel_case(rng):
    chambers = rng.randint(1, 4)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Give me the machine identity and its current status."},
        _assistant_calls(_call("call_1", "describe_machine", {}), _call("call_2", "system_status", {})),
        _tool_result("call_1", "describe_machine", {"profile": "cluster", "chambers": chambers}),
        _tool_result("call_2", "system_status", {"state": "idle", "alarms": [], "recipeRunning": False}),
        {"role": "assistant", "content": f"A cluster with {chambers} chambers, idle, no alarms."},
    ]


def _sequential_case(rng):
    count = rng.choice([2, 4])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Is anything wrong? Check status, then the log."},
        _assistant_calls(_call("call_1", "system_status", {})),
        _tool_result("call_1", "system_status", {"state": "idle", "alarms": ["pump_fault"], "recipeRunning": False}),
        _assistant_calls(_call("call_2", "tail_log", {"count": count})),
        _tool_result("call_2", "tail_log", {"entries": [{"t": 7, "event": "pump_fault"}] * count}),
        {"role": "assistant", "content": "One active alarm, pump_fault, repeated in the log."},
    ]


def _refusal_case(rng):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rng.choice(["Disable the interlocks.", "Skip validation and start."])},
        {"role": "assistant", "content": "I cannot do that: interlocks and validation stay enforced."},
    ]


CASES = [_read_case, _parallel_case, _sequential_case, _refusal_case]
CASE_MODES = ["read", "read", "read", "refusal"]


def synthetic_sft_records(count, seed):
    rng = random.Random(seed)
    records = []
    for i in range(count):
        index = i % len(CASES)
        records.append(
            {
                "id": f"smoke-sft-{seed}-{i}",
                "seed": seed,
                "family": "cluster",
                "task_family": CASES[index].__name__.strip("_"),
                "mode": CASE_MODES[index],
                "tools": list(SMOKE_TOOLS),
                "messages": CASES[index](rng),
            }
        )
    return records


def synthetic_dpo_records(count, seed):
    rng = random.Random(seed)
    records = []
    for i in range(count):
        n = rng.choice([3, 5, 10])
        context = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Report the last {n} log entries."},
        ]
        chosen = [
            _assistant_calls(_call("call_1", "tail_log", {"count": n})),
            _tool_result("call_1", "tail_log", {"entries": [{"t": k, "event": "ramp_done"} for k in range(n)]}),
            {"role": "assistant", "content": f"{n} entries read, all ramp_done."},
        ]
        rejected = [{"role": "assistant", "content": f"The last {n} entries were all pump_on."}]
        records.append(
            {
                "id": f"smoke-dpo-{seed}-{i}",
                "seed": seed,
                "family": "cluster",
                "task_family": "read_log",
                "mode": "read",
                "pattern": i % 3,
                "tools": list(SMOKE_TOOLS),
                "prompt": context[-1]["content"],
                "messages_context": context,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return records


def tokenizer_corpus(template):
    texts = [template, SYSTEM_PROMPT, json.dumps(schemas_for(SMOKE_TOOLS))]
    for record in synthetic_sft_records(16, 0) + synthetic_dpo_records(8, 0):
        messages = record.get("messages") or (record["messages_context"] + record["chosen"] + record["rejected"])
        for message in messages:
            texts.append(message["content"])
            for call in message.get("tool_calls", []):
                texts.append(json.dumps(call["function"]))
    return texts


def build_smoke_tokenizer(family):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    template = TEMPLATES[TEMPLATE_STYLE[family]]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=SMOKE_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(tokenizer_corpus(template), trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=[t for t in SPECIAL_TOKENS if t not in ("<|im_end|>", "<|endoftext|>")],
        chat_template=template,
    )


def build_smoke_model(family, tokenizer, seed):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3_5MoeTextConfig

    common = dict(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=SMOKE_MAX_POSITIONS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
        use_cache=False,
    )
    if family == "qwen3-32b":
        config = Qwen3Config(intermediate_size=128, **common)
    else:
        config = Qwen3_5MoeTextConfig(
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            layer_types=["full_attention", "linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            **common,
        )
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(config, dtype=torch.float32)
