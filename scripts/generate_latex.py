
import ast
import os
import re

# Map Python operators to LaTeX
OP_MAP = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '\\cdot ',
    ast.Div: '/',
    ast.USub: '-',
    ast.Eq: '=',
    ast.LtE: '\\leq',
    ast.GtE: '\\geq',
}

def to_latex(node):
    """Recursively convert AST node to LaTeX string."""
    if isinstance(node, ast.Name):
        name = node.id
        
        # Greek letters
        greeks = ['lam', 'mu', 'gamma', 'beta', 'psi', 'rho', 'tau', 'eps', 'omega', 'theta', 'alpha', 'delta']
        if name in greeks:
             return f"\\{name}"
        for g in greeks:
            if name.startswith(g + '_'):
                suffix = name[len(g)+1:]
                return f"\\{g}_{{{suffix.replace('_', '\\_')}}}"

        if name == 'z':
            return "0"
        if name == 'z_llp':
            return "z_{LLP}"
            
        return name.replace('_', '\\_')

    elif isinstance(node, ast.Constant):
        return str(node.value)

    elif isinstance(node, ast.BinOp):
        left = to_latex(node.left)
        right = to_latex(node.right)
        op = OP_MAP.get(type(node.op))
        
        if isinstance(node.op, ast.Div):
            return f"\\frac{{{left}}}{{{right}}}"
        
        # Parentheses for precedence
        if isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Mult, ast.Div)):
             pass 
        elif isinstance(node.op, ast.Mult) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, (ast.Add, ast.Sub)):
            left = f"({left})"
        elif isinstance(node.op, ast.Mult) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
            right = f"({right})"
            
        return f"{left} {op} {right}"

    elif isinstance(node, ast.UnaryOp):
        operand = to_latex(node.operand)
        op = OP_MAP.get(type(node.op), '')
        return f"{op}{operand}"

    elif isinstance(node, ast.Compare):
        left = to_latex(node.left)
        ops = [OP_MAP.get(type(op), '?') for op in node.ops]
        comparators = [to_latex(comp) for comp in node.comparators]
        
        result = left
        for op, comp in zip(ops, comparators):
            result += f" {op} {comp}"
        return result

    elif isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == 'Sum':
            if len(node.args) >= 2:
                # Handle list of indices: Sum([i,j], ...)
                idx_arg = node.args[0]
                if isinstance(idx_arg, ast.List):
                    indices = [to_latex(e) for e in idx_arg.elts]
                    index_str = ",".join(indices)
                else:
                    index_str = to_latex(idx_arg)
                    
                expr = to_latex(node.args[1])
                return f"\\sum_{{{index_str}}} ({expr})"
            return "\\sum(?)"
        
        elif func_name == 'Number': 
             if len(node.args) > 0:
                 return to_latex(node.args[0])
             return ""

        else:
             args = [to_latex(a) for a in node.args]
             return f"\\text{{{func_name}}}({', '.join(args)})"

    elif isinstance(node, ast.Subscript):
        value = to_latex(node.value)
        if isinstance(node.slice, ast.Tuple):
            indices = [to_latex(e) for e in node.slice.elts]
            idx_str = ','.join(indices)
        else:
            idx_str = to_latex(node.slice)
        return f"{value}_{{{idx_str}}}"
    
    elif isinstance(node, ast.Tuple):
        elts = [to_latex(e) for e in node.elts]
        return f"({', '.join(elts)})"
        
    elif isinstance(node, ast.List):
        elts = [to_latex(e) for e in node.elts]
        return f"[{', '.join(elts)}]"

    return ""

def extract_model_info(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    variables = {'ULP': [], 'LLP': []}
    equations = {'ULP': [], 'LLP': []}
    stabilization = []
    
    current_section = 'ULP' # Default start (or irrelevant)

    for node in ast.walk(tree):
        # Detect simple Variable definitions
        if isinstance(node, ast.Assign):
            # Check for Variable definitions: var = Variable(...)
            if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'Variable':
                var_name = node.targets[0].id
                var_type = "FREE"
                domain = ""
                
                # Extract args/keywords
                for kw in node.value.keywords:
                    if kw.arg == 'type':
                        if 'POSITIVE' in ast.dump(kw.value):
                            var_type = 'POSITIVE'
                    if kw.arg == 'domain':
                        domain = to_latex(kw.value)

                entry = {'name': var_name, 'domain': domain, 'type': var_type}
                
                # Heuristic to classify variable based on name
                if var_name in ['Q_offer', 'tau_imp', 'tau_exp']:
                    variables['ULP'].append(entry)
                elif var_name in ['x', 'x_dem', 'lam', 'mu', 'gamma', 'beta_dem', 'psi_dem', 'z_llp']:
                    variables['LLP'].append(entry)

            # Check for Equation assignments: eq_name[...] = ...
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if isinstance(target.value, ast.Name):
                    eq_name = target.value.id
                    if eq_name.startswith('eq_'):
                        # Determine domain
                        if isinstance(target.slice, ast.Tuple):
                            indices = [to_latex(e) for e in target.slice.elts]
                        else:
                            indices = [to_latex(target.slice)]
                        idx_str = ', '.join(indices) if indices[0] != '...' else ''
                        
                        latex_expr = to_latex(node.value)
                        equations['LLP'].append({'name': eq_name, 'domain': idx_str, 'latex': latex_expr})

            # Check for penalty term assignments: pen_... = ...
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.startswith('pen_'):
                    latex_expr = to_latex(node.value)
                    stabilization.append({'name': name, 'latex': latex_expr})
                    
                # Capture ULP Objective
                if name == 'obj_welfare':
                     latex_expr = to_latex(node.value)
                     equations['ULP'].append({'name': 'Objective (Welfare)', 'domain': '', 'latex': latex_expr})

    return variables, equations, stabilization

def generate_markdown(out_file, variables, equations, stabilization):
    with open(out_file, 'w') as f:
        f.write("# Model Equations\n\n")
        
        # --- ULP Section ---
        f.write("## Upper Level Problem (ULP)\n\n")
        f.write("### Variables\n")
        for v in variables['ULP']:
            if v['type'] == 'POSITIVE':
                 domain = f" \\in {v['domain']}" if v['domain'] else ""
                 f.write(f"$$ \\{to_latex(ast.Name(id=v['name']))} \\geq 0 \\quad \\forall {domain} $$\n")
        
        f.write("\n### Objectives\n")
        for eq in equations['ULP']:
            f.write(f"**{eq['name']}**:\n")
            f.write(f"$$ {eq['latex']} $$\n\n")

        # --- LLP Section ---
        f.write("\n## Lower Level Problem (LLP)\n\n")
        f.write("### Variables\n")
        for v in variables['LLP']:
            if v['type'] == 'POSITIVE':
                 domain = f" \\in {v['domain']}" if v['domain'] else ""
                 f.write(f"$$ \\{to_latex(ast.Name(id=v['name']))} \\geq 0 \\quad \\forall {domain} $$\n")

        f.write("\n### Equations\n")
        for eq in equations['LLP']:
            name_clean = eq['name'].replace('_', ' ')
            domain = f", \\forall {eq['domain']}" if eq['domain'] else ""
            f.write(f"**{name_clean}** ({domain}):\n")
            f.write(f"$$ {eq['latex']} $$\n\n")

        # --- Stabilization Section ---
        f.write("\n## Numerical Stabilization\n\n")
        f.write("The following penalty terms are used in the ULP objective or LLP regularization:\n\n")
        for term in stabilization:
            name_clean = term['name'].replace('_', ' ')
            f.write(f"**{name_clean}**:\n")
            f.write(f"$$ {term['latex']} $$\n\n")

def generate_tex(out_file, variables, equations, stabilization):
    with open(out_file, 'w') as f:
        # Preamble (minimal)
        f.write("% Model Equations Auto-Generated\n")
        f.write("\\section{Upper Level Problem (ULP)}\n\n")
        
        f.write("\\subsection{Variables}\n")
        f.write("\\begin{itemize}\n")
        for v in variables['ULP']:
            if v['type'] == 'POSITIVE':
                 domain = f" \\in {v['domain']}" if v['domain'] else ""
                 f.write(f"    \\item \\( \\{to_latex(ast.Name(id=v['name']))} \\geq 0 \\quad \\forall {domain} \\)\n")
        f.write("\\end{itemize}\n")

        f.write("\n\\subsection{Objectives}\n")
        for eq in equations['ULP']:
            f.write(f"\\subsubsection*{{{eq['name']}}}\n")
            f.write("\\begin{equation}\n")
            f.write(f"    {eq['latex']}\n")
            f.write("\\end{equation}\n\n")

        # --- LLP Section ---
        f.write("\n\\section{Lower Level Problem (LLP)}\n\n")
        
        f.write("\\subsection{Variables}\n")
        f.write("\\begin{itemize}\n")
        for v in variables['LLP']:
            if v['type'] == 'POSITIVE':
                 domain = f" \\in {v['domain']}" if v['domain'] else ""
                 f.write(f"    \\item \\( \\{to_latex(ast.Name(id=v['name']))} \\geq 0 \\quad \\forall {domain} \\)\n")
        f.write("\\end{itemize}\n")

        f.write("\n\\subsection{Equations}\n")
        for eq in equations['LLP']:
            name_clean = eq['name'].replace('_', ' ')
            domain = f", \\forall {eq['domain']}" if eq['domain'] else ""
            f.write(f"\\subsubsection*{{{name_clean} ({domain})}}\n")
            f.write("\\begin{equation}\n")
            f.write(f"    {eq['latex']}\n")
            f.write("\\end{equation}\n\n")

        # --- Stabilization Section ---
        f.write("\n\\section{Numerical Stabilization}\n")
        f.write("The following penalty terms are used in the ULP objective or LLP regularization:\n\n")
        for term in stabilization:
            name_clean = term['name'].replace('_', ' ')
            f.write(f"\\subsubsection*{{{name_clean}}}\n")
            f.write("\\begin{equation}\n")
            f.write(f"    {term['latex']}\n")
            f.write("\\end{equation}\n\n")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_file = os.path.join(base_dir, 'src', 'solargeorisk_extension', 'model.py')
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    out_md = os.path.join(output_dir, 'model_equations.md')
    out_tex = os.path.join(output_dir, 'model_equations.tex')

    if not os.path.exists(src_file):
        print(f"Error: Could not find {src_file}")
        return

    print(f"Parsing {src_file}...")
    variables, equations, stabilization = extract_model_info(src_file)
    
    generate_markdown(out_md, variables, equations, stabilization)
    print(f"Markdown written to {out_md}")
    
    generate_tex(out_tex, variables, equations, stabilization)
    print(f"LaTeX source written to {out_tex}")

    # Also write to Overleaf directory
    overleaf_dir = os.path.join(base_dir, 'overleaf')
    if os.path.exists(overleaf_dir):
        out_overleaf = os.path.join(overleaf_dir, 'model_equations.tex')
        generate_tex(out_overleaf, variables, equations, stabilization)
        print(f"LaTeX source also written to {out_overleaf}")

if __name__ == "__main__":
    main()
