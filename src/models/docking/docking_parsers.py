"""Docking output parsers."""
import re

def parse_vina_output_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    scores = []
    for line in content.split('\n'):
        if 'REMARK VINA RESULT:' in line:
            match = re.search(r'REMARK VINA RESULT:\s+([-\d.]+)', line)
            if match:
                scores.append(float(match.group(1)))
    
    return {
        'best_score': min(scores) if scores else None,
        'all_scores': scores,
        'num_poses': len(scores)
    }
