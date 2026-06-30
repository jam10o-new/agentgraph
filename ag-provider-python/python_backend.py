import argparse
import json
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def emit(line: str):
    print(line, flush=True)


def emit_done():
    emit("D")


def emit_error(msg: str):
    emit(f"X:{msg}")


def get_model():
    model_path = os.environ.get("AG_PROVIDER_PYTHON_MODEL")
    if not model_path:
        emit_error("AG_PROVIDER_PYTHON_MODEL not set")
        sys.exit(1)
    try:
        from llama_cpp import Llama

        model = Llama(model_path=model_path, n_ctx=32768, verbose=False)
        return model
    except ImportError:
        emit_error("llama-cpp-python not installed; pip install llama-cpp-python")
        sys.exit(1)


def handle_chat(model, request):
    messages = request.get("messages", [])
    tools = request.get("tools")
    tool_choice = request.get("tool_choice")
    constraint = request.get("constraint")
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 1024)
    top_p = request.get("top_p", 0.95)
    enable_thinking = request.get("enable_thinking", False)

    # Build kwargs for create_chat_completion
    kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
    )

    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    if constraint:
        # Grammar/JSON mode: use response_format
        schema = constraint.get("schema")
        grammar = constraint.get("grammar")
        regex = constraint.get("regex")
        if schema:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}
        elif grammar:
            kwargs["grammar"] = grammar
        elif regex:
            kwargs["grammar"] = regex  # llama.cpp uses GBNF

    try:
        stream = model.create_chat_completion(**kwargs)
    except Exception as e:
        emit_error(str(e))
        return

    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            emit(f"T:{content}")
        reasoning = delta.get("reasoning", "")
        if enable_thinking and reasoning:
            emit(f"R:{reasoning}")
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                emit(f"C:{json.dumps({'id': tc.get('id', ''), 'name': fn.get('name', ''), 'arguments': fn.get('arguments', '')}, ensure_ascii=False)}")

    emit_done()


def handle_embed(model, request):
    text = request.get("text", "")
    if not text:
        emit_error("embed: empty text")
        return

    try:
        result = model.create_embedding(input=[text])
    except Exception as e:
        emit_error(f"embed error: {e}")
        return

    data = result.get("data", [])
    if data:
        embedding = data[0].get("embedding", [])
        emit(f"E:{json.dumps(embedding, ensure_ascii=False)}")
    else:
        emit_error("embed: no data in response")
        return

    emit_done()


def handle_info(model):
    info = {
        "name": "python",
        "model": os.environ.get("AG_PROVIDER_PYTHON_MODEL"),
        "max_seq_len": None,
        "supports_tools": True,
        "supports_embeddings": True,
        "supports_constraints": True,
        "supports_modalities": ["text", "image"],
    }
    emit(f"I:{json.dumps(info, ensure_ascii=False)}")
    emit_done()


def handle_health(model):
    emit("H:true")
    emit_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["chat", "embed", "health", "info"])
    parser.add_argument("--model")
    args = parser.parse_args()

    # Read JSON request from stdin
    raw = sys.stdin.read()
    if not raw.strip():
        emit_error("no input on stdin")
        return

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        emit_error(f"invalid json: {e}")
        return

    if args.mode in ("chat", "embed", "health", "info"):
        model = get_model()
        if args.mode == "chat":
            handle_chat(model, request)
        elif args.mode == "embed":
            handle_embed(model, request)
        elif args.mode == "health":
            handle_health(model)
        elif args.mode == "info":
            handle_info(model)


if __name__ == "__main__":
    main()
