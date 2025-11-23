from agentify.core.tool import Tool
import ast
import operator as op


calculate_expression_schema = {
    "name": "calculate_expression",
    "description": "Evalúa una expresión matemática segura y devuelve el resultado.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expresión matemática a calcular, por ejemplo '2 + 2 * (3 - 1)'.",
            }
        },
        "required": ["expression"],
    },
}


_allowed_ops = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _allowed_ops[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        return _allowed_ops[type(node.op)](operand)
    raise ValueError(f"Operador no permitido: {node}")


def calculate_expression(expression: str):
    try:
        tree = ast.parse(expression, mode="eval").body
        result = _eval_node(tree)
        return {"result": result}
    except Exception as e:
        return {"error": f"Expresión inválida: {e}"}


calculate_expression_tool = Tool(calculate_expression_schema, calculate_expression)
