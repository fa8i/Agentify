from agentify.core.tool import Tool
import ast
import operator as op


class CalculatorTool(Tool):
    """Tool for evaluating safe mathematical expressions."""

    def __init__(self):
        self._allowed_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.Mod: op.mod,
            ast.UAdd: op.pos,
            ast.USub: op.neg,
        }

        schema = {
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
        super().__init__(schema, self._calculate_expression)

    def _eval_node(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._allowed_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self._allowed_ops[type(node.op)](operand)
        raise ValueError(f"Operador no permitido: {node}")

    def _calculate_expression(self, expression: str):
        try:
            tree = ast.parse(expression, mode="eval").body
            result = self._eval_node(tree)
            return {"result": result}
        except Exception as e:
            return {"error": f"Expresión inválida: {e}"}
